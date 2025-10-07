import asyncio
import logging
import re
import time
from collections import defaultdict
from functools import partial
from concurrent.futures import Future

import chz
import datasets
import numpy
import tinker
import torch
from tinker import ModelInput, SamplingClient, types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

DOUBLE_COT_SYSTEM_PROMPT = '''/no_think
Answer the provided questions after providing a few sentences of reasoning for each question. Your answer must only contain a single integer with no additional symbols or phrases. Do not output anything other than a single integer between the <answer> and </answer> tags.
Respond in the following format:
<reasoning1>
...
</reasoning1>

<answer1>
...
</answer1>

<reasoning2>
...
</reasoning2>

<answer2>
...
</answer2>
'''

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "checkpoints/length_penalty"
    model_name: str = "Qwen/Qwen3-32B"
    batch_size: int = 128
    group_size: int = 16
    learning_rate: float = 4e-5
    max_length: int = 32768
    lora_rank: int = 32
    save_every: int = 50
    val_every: int = 5
    max_tokens: int = 2048
    num_epochs: int = 20
    reward: str = "length_penalty"

def extract_xml(text, opening_tag, closing_tag):
    content = text.split(opening_tag.strip())[-1]
    content = content.split(closing_tag.strip())[0]
    return content.strip()

def find_last_subsequence(lst, sub):
    n, m = len(lst), len(sub)
    if m == 0 or m > n:
        return -1

    # iterate backwards so the first match we find is the last occurrence
    for i in range(n - m, -1, -1):
        if lst[i:i + m] == sub:
            return i
    return -1

def get_base_reward(response: str, answer: str) -> dict:
    format_reward = 0
    correctness_reward = 0
    
    format_pattern = fr"<reasoning1>\s*.*?\s*</reasoning1>\s*<answer1>\s*.*?\s*</answer1>\s*<reasoning2>\s*.*?\s*</reasoning2>\s*<answer2>\s*.*?\s*</answer2>$"

    reasoning_trace_1 = str()
    reasoning_trace_2 = str()

    if re.search(format_pattern, response):
        format_reward += 0.2

        parsed_answer = extract_xml(response, '<answer2>', '</answer2>')
        reasoning_trace_1 = extract_xml(response, '<reasoning1>', '</reasoning1>')
        reasoning_trace_2 = extract_xml(response, '<reasoning2>', '</reasoning2>')

        if parsed_answer == answer:
            correctness_reward += 0.8
    return format_reward, correctness_reward, reasoning_trace_1, reasoning_trace_2

async def get_length_penalty_reward(tokens: list[int], response: str, answer: str) -> dict:
    format_reward, correctness_reward, reasoning_trace_1, reasoning_trace_2 = get_base_reward(response, answer)
    length_reward = 0

    if correctness_reward > 0:
        length_ratio = len(reasoning_trace_1) / (len(reasoning_trace_1) + len(reasoning_trace_2))
        length_reward += length_ratio * 0.5

    return {
        "score": format_reward + correctness_reward + length_reward,
        "format_reward": format_reward,
        "correctness_reward": correctness_reward,
        "length_reward": length_reward,
        "trace1_length": len(reasoning_trace_1),
        "trace2_length": len(reasoning_trace_2),
    }

async def get_logprob_reward(base_sampling_client: SamplingClient, tokenizer, tokens: list[int], response: str, answer: str) -> dict:
    format_reward, correctness_reward, reasoning_trace_1, reasoning_trace_2 = get_base_reward(response, answer)
    logprob_reward = 0

    if correctness_reward > 0:
        base_logprobs = await base_sampling_client.compute_logprobs_async(ModelInput.from_ints(tokens))

        # we know the answer is correct so just look for it
        tokenized_answer = tokenizer.encode(answer)
        answer_start = find_last_subsequence(tokens, tokenized_answer)
        answer_logprobs = base_logprobs[answer_start:answer_start + len(tokenized_answer)]
        mean_logprob = sum(answer_logprobs) / len(answer_logprobs)
        logprob_reward -= 0.5 * numpy.exp(mean_logprob)

    return {
        "score": format_reward + correctness_reward + logprob_reward,
        "format_reward": format_reward,
        "correctness_reward": correctness_reward,
        "logprob_reward": logprob_reward,
        "trace1_length": len(reasoning_trace_1),
        "trace2_length": len(reasoning_trace_2),
    }

async def get_attn_reward(sampling_client: SamplingClient, tokenizer, tokens: list[int], response: str, answer: str) -> dict:
    format_reward, correctness_reward, reasoning_trace_1, reasoning_trace_2 = get_base_reward(response, answer)
    attn_reward = 0

    if correctness_reward > 0:
        orig_logprobs = await sampling_client.compute_logprobs_async(ModelInput.from_ints(tokens))

        # we know formatted
        # find whats between reasoning1 and /reasoning1
        # cut that out (replace with tokenize(...))

        reasoning1_start_pos = max(find_last_subsequence(tokens, tokenizer.encode(pattern)) + len(tokenizer.encode(pattern)) for pattern in ['<reasoning1', '\n<reasoning1', '.<reasoning1']) + 1
        reasoning1_end_pos = max(find_last_subsequence(tokens, tokenizer.encode(pattern)) for pattern in ['</reasoning1', '\n</reasoning1', '.</reasoning1'])
        spliced_tokens = tokens[:reasoning1_start_pos] + tokenizer.encode('...\n') + tokens[reasoning1_end_pos:]
        masked_logprobs = await sampling_client.compute_logprobs_async(ModelInput.from_ints(spliced_tokens))

        # we know the answer is correct so just look for it
        tokenized_answer = tokenizer.encode(answer)

        orig_answer_start = find_last_subsequence(tokens, tokenized_answer)
        orig_answer_logprobs = orig_logprobs[orig_answer_start:orig_answer_start + len(tokenized_answer)]
        orig_mean_logprob = sum(orig_answer_logprobs) / len(orig_answer_logprobs)

        masked_answer_start = find_last_subsequence(spliced_tokens, tokenized_answer)
        masked_answer_logprobs = masked_logprobs[masked_answer_start:masked_answer_start + len(tokenized_answer)]
        masked_mean_logprob = sum(masked_answer_logprobs) / len(masked_answer_logprobs)

        # positive if the model did better seeing the q1 trace
        logprob_diff = orig_mean_logprob - masked_mean_logprob

        alpha = 2
        logprob_diff = min(alpha, logprob_diff)
        attn_reward += (numpy.exp(logprob_diff) - 1) / (alpha - 1)


    return {
        "score": format_reward + correctness_reward + attn_reward,
        "format_reward": format_reward,
        "correctness_reward": correctness_reward,
        "attn_reward": attn_reward,
        "trace1_length": len(reasoning_trace_1),
        "trace2_length": len(reasoning_trace_2),
    }


async def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project='tinker_encoded_reasoning',
        wandb_name=config.reward,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load GSM8K dataset
    logger.info("Loading dataset...")
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    N_train = 7473
    N_val = 512
    train_dataset = datasets.Dataset.from_list(
    [
        {
            'question': f'''Question 1:\n{example1['question']}\n\nQuestion 2:\n{example2['question']}''',
            'answer': example2['answer'].split('####')[-1].strip(),
        }
        for example1, example2 in zip(
            dataset["train"].select([i%N_train for i in range(0, N_train*2, 2)]), 
            dataset["train"].select([i%N_train for i in range(1, N_train*2, 2)])
        )
    ])
    train_dataset = datasets.concatenate_datasets([train_dataset] * config.num_epochs).shuffle()
    val_dataset = datasets.Dataset.from_list(
    [
        {
            'question': f'''Question 1:\n{example1['question']}\n\nQuestion 2:\n{example2['question']}''',
            'answer': example2['answer'].split('####')[-1].strip(),
        }
        for example1, example2 in zip(
            dataset["test"].select([i%N_val for i in range(0, N_val*2, 2)]), 
            dataset["test"].select([i%N_val for i in range(1, N_val*2, 2)])
        )
    ])

    convo_prefix = [
        {
            "role": "system",
            "content": DOUBLE_COT_SYSTEM_PROMPT,
        }
    ]

    n_train_batches = len(train_dataset) // config.batch_size
    n_val_batches = len(val_dataset) // config.batch_size

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    # Optimizer step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = await service_client.create_training_client_from_state_async(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    logger.info(f"Training for {n_train_batches} batches")

    #  Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        # Setup metrics for logging
        t_start = time.time()
        step = batch_idx
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get training batch and convert to datums online
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)
        # Set up sampling parameters

        if config.reward == "length_penalty":
            get_reward = get_length_penalty_reward
        elif config.reward == "logprob":
            base_sampling_client = service_client.create_sampling_client(base_model=config.model_name)
            get_reward = partial(get_logprob_reward, base_sampling_client, tokenizer)
        elif config.reward == "attn":
            get_reward = partial(get_attn_reward, sampling_client, tokenizer)
        else:
            raise NotImplementedError(config.reward)

        if step % config.val_every == 0:
            val_metrics = defaultdict(list)

            reward_tasks = []
            for val_batch_idx in range(n_val_batches):
                val_batch_start = batch_idx * config.batch_size
                val_batch_end = min((batch_idx + 1) * config.batch_size, len(val_dataset))
                val_batch_rows = val_dataset.select(range(val_batch_start, val_batch_end))

                val_batch_futures: list[list[Future[types.SampleResponse]]] = []
                val_batch_inputs: list[ModelInput] = []
                for question in val_batch_rows["question"]:
                    convo = [
                        *convo_prefix,
                        {"role": "user", "content": question},
                    ]
                    model_input = renderer.build_generation_prompt(convo)
                    prompt_tokens = model_input.to_ints()

                    # Generate response
                    sample_futures: list[Future[types.SampleResponse]] = []
                    sample_futures.append(
                        sampling_client.sample(
                            prompt=model_input,
                            num_samples=1,
                            sampling_params=sampling_params,
                        )
                    )

                    val_batch_futures.append(sample_futures)
                    val_batch_inputs.append(model_input)

                for sample_futures, model_input, answer in zip(
                    val_batch_futures, val_batch_inputs, val_batch_rows["answer"]
                ):
                    prompt_tokens = model_input.to_ints()
                    for future in sample_futures:
                        sample_result = future.result()
                        sampled_tokens = sample_result.sequences[0].tokens
                        parsed_message, _ = renderer.parse_response(sampled_tokens)

                        if val_batch_idx == 0:
                            logger.info(tokenizer.decode(prompt_tokens) + parsed_message["content"])
                        reward_tasks.append(get_reward(prompt_tokens + sampled_tokens, parsed_message["content"], answer))

                        
            rewards = await asyncio.gather(*reward_tasks)
            for reward in rewards:
                for metric, value in reward.items():
                    val_metrics[metric].append(value)

            for metric, values in val_metrics.items():
                metrics[f"val/{metric}/mean"] = sum(values) / len(values)

        training_datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_futures: list[list[Future[types.SampleResponse]]] = []
        batch_inputs: list[ModelInput] = []
        batch_metrics = defaultdict(list)
        for question in batch_rows["question"]:
            convo = [
                *convo_prefix,
                {"role": "user", "content": question},
            ]
            model_input = renderer.build_generation_prompt(convo)
            prompt_tokens = model_input.to_ints()

            # Generate response
            sample_futures: list[Future[types.SampleResponse]] = []
            for _ in range(config.group_size):
                sample_futures.append(
                    sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                )

            batch_futures.append(sample_futures)
            batch_inputs.append(model_input)

        for sample_futures, model_input, answer in zip(
            batch_futures, batch_inputs, batch_rows["answer"]
        ):
            reward_tasks = []

            group_rewards: list[float] = []
            group_tokens: list[list[int]] = []
            group_logprobs: list[list[float]] = []
            group_ob_lens: list[int] = []
            prompt_tokens = model_input.to_ints()
            for future in sample_futures:
                sample_result = future.result()
                sampled_tokens = sample_result.sequences[0].tokens
                sampled_logprobs = sample_result.sequences[0].logprobs
                assert sampled_logprobs is not None

                all_tokens = prompt_tokens + sampled_tokens
                group_tokens.append(all_tokens)
                group_ob_lens.append(len(prompt_tokens) - 1)
                group_logprobs.append(sampled_logprobs)

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                reward_tasks.append(get_reward(prompt_tokens + sampled_tokens, parsed_message["content"], answer))

            rewards = await asyncio.gather(*reward_tasks)
            for reward in rewards:
                group_rewards.append(reward["score"])
                for metric, value in reward.items():
                    batch_metrics[metric].append(value)

            advantages = [
                reward - (sum(group_rewards) / len(group_rewards)) for reward in group_rewards
            ]
            batch_rewards.append(sum(group_rewards) / len(group_rewards))

            # check if all advantages are zero
            if all(advantage == 0.0 for advantage in advantages):
                # Skip question because all advantages are the same
                continue

            for tokens, logprob, advantage, ob_len in zip(
                group_tokens, group_logprobs, advantages, group_ob_lens
            ):
                input_tokens = tokens[:-1]
                input_tokens = [int(token) for token in input_tokens]
                target_tokens = tokens[1:]
                all_logprobs = [0.0] * ob_len + logprob
                all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
                assert (
                    len(input_tokens)
                    == len(target_tokens)
                    == len(all_logprobs)
                    == len(all_advantages)
                ), (
                    f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, len(all_logprobs): {len(all_logprobs)}, len(all_advantages): {len(all_advantages)}"
                )
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)

        # Training step
        fwd_bwd_future = training_client.forward_backward(
            training_datums, loss_fn="ppo"
        )
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Log metrics[]
        metrics["time/total"] = time.time() - t_start
        for metric, values in batch_metrics.items():
            metrics[f"reward/{metric}/mean"] = sum(values) / len(values)
        ml_logger.log_metrics(metrics, step=batch_idx)

        # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    def _main(config: Config):
        asyncio.run(main(config))
    chz.nested_entrypoint(_main)
