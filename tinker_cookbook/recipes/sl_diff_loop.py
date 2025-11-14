"""
Minimal supervised fine-tuning script without abstractions.
Uses existing modules but with a simple, flat training loop.
"""

import logging
import time

import chz
import datasets
import numpy as np
import tinker
from tinker import Datum, TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
import torch

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

ENCODED_SYSTEM_PROMPT = '''/no_think
You will be given two questions and must answer both questions. You can provide a few sentences of reasoning but only pertaining to the first question. You are not allowed to discuss the second question in your reasoning. 

Your answer for each question must only contain a single integer with no additional symbols or phrases. Do not output anything other than a single integer between the <answer> and </answer> tags.
<reasoning>
...
</reasoning>

<answer1>
...
</answer1>

<answer2>
...
</answer2>
'''

BLIND_SINGLE_SYSTEM_PROMPT = '''/no_think
Answer the provided question directly without providing any additional reasoning or output. Your answer for each question must only contain a single integer with no additional symbols or phrases. Do not output anything other than a single integer between the <answer> and </answer> tags.
Respond in the following format.
<answer>
...
</answer>
'''

def find_last_subseq_start(strings, target):
    n = len(strings)
    target_len = len(target)
    
    # Iterate from the end backwards to find the *last* subsequence
    for start in range(n - 1, -1, -1):
        concat_pos = 0
        for end in range(start, n):
            s = strings[end]
            # If this string would go beyond target, stop
            if concat_pos + len(s) > target_len:
                break
            # Check if current chunk matches the target at this position
            if target[concat_pos:concat_pos+len(s)] != s:
                break  # mismatch — stop searching from this start
            concat_pos += len(s)
            if concat_pos == target_len:
                return start  # exact match
        # otherwise continue to next start index
    return -1  # not found

@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "checkpoints/sft"
    model_name: str = "Qwen/Qwen3-8B"
    batch_size: int = 64 # 128
    learning_rate: float = 1e-4
    max_length: int = 32768
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    lora_rank: int = 32
    save_every: int = 50
    num_epochs: int = 50
    n: int = 5


# source .bashrc; conda activate inspect; cd tinker-cookbook; python tinker_cookbook/recipes/sl_diff_loop.py log_path=checkpoints/sft_32b_mult_4_diff model_name=Qwen/Qwen3-32B n=4
def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project='tinker_encoded_reasoning',
        wandb_name=f'n={config.n} sft ' + config.model_name.split('/')[-1],
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load No Robots dataset
    logger.info("Loading dataset...")
    import random
    random.seed(12345)
    N_train = 5000 * config.num_epochs
    n = config.n

    train_samples = []
    train_random_samples = []

    for _ in range(N_train * config.num_epochs):
        a1, b1 = random.randint(10 ** (n-1), 10 ** n - 1), random.randint(10 ** (n-1), 10 ** n - 1)
        train_samples.append({"messages": [
            {"role": "system", "content": BLIND_SINGLE_SYSTEM_PROMPT},
            {"role": "user", "content": f'''What is {a1} times {b1}?'''},
            {"role": "assistant", "content": f'''<answer>\n{a1*b1}\n</answer>'''}
        ]})
        train_random_samples.append({"messages": [
            {"role": "system", "content": BLIND_SINGLE_SYSTEM_PROMPT},
            {"role": "user", "content": f'''What is {a1} times {b1}?  {"".join(str(i) for i in random.choices(range(10), k=random.randint(50, 50)))}'''},
            {"role": "assistant", "content": f'''<answer>\n{a1*b1}\n</answer>'''}
        ]})

    train_dataset = datasets.Dataset.from_list(train_samples)
    train_random_dataset = datasets.Dataset.from_list(train_random_samples)
    n_train_batches = len(train_dataset) // config.batch_size
    logger.info(f"Train batches: {n_train_batches}")

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Check for resuming
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    # Training loop (single epoch)
    logger.info(f"Training for {n_train_batches} steps")

    # Shuffle dataset
    train_dataset = train_dataset.shuffle(seed=0)
    train_random_dataset = train_random_dataset.shuffle(seed=0)

    for batch_idx in range(start_batch, n_train_batches):
        start_time = time.time()
        step = batch_idx
        metrics = {}

        # Save checkpoint
        if step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Linear learning rate schedule
        lr_mult = max(0.0, 1.0 - step / n_train_batches)
        current_lr = config.learning_rate * lr_mult
        adam_params = tinker.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

        # Get training batch and convert to datums online
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))
        batch_random_rows = train_random_dataset.select(range(batch_start, batch_end))

        batch = [
            conversation_to_datum(
                row["messages"],  # type: ignore
                renderer,
                config.max_length,
                config.train_on_what,
            )
            for row in datasets.concatenate_datasets([batch_rows, batch_random_rows])
        ]

        # Training step
        def loss_fn(data: list[Datum], logprobs: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
            loss = 0
            for i in range(len(data) // 2, len(data)):
                seq_data = data[i]
                seq_logprobs = logprobs[i]

                base_data = data[i - len(data) // 2]
                base_logprobs = logprobs[i - len(data) // 2]

                resp_tokens = np.array(seq_data.loss_fn_inputs['weights'].data) * np.array(seq_data.model_input.chunks[0].tokens)
                resp_mask = resp_tokens > 0
                answer_mask = (resp_tokens >= 15) & (resp_tokens <= 24)

                base_resp_tokens = np.array(base_data.loss_fn_inputs['weights'].data) * np.array(base_data.model_input.chunks[0].tokens)
                base_resp_mask = base_resp_tokens > 0
                base_answer_mask = (base_resp_tokens >= 15) & (base_resp_tokens <= 24)

                loss += -(seq_logprobs[answer_mask] - base_logprobs.detach()[base_answer_mask]).sum() - base_logprobs[base_resp_mask & ~base_answer_mask].sum() - seq_logprobs[resp_mask & ~answer_mask].sum()

            return loss, {"logprob_diff_loss": loss.item()}
        fwd_bwd_future = training_client.forward_backward_custom(batch, loss_fn)
        optim_step_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Compute train metrics
        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch]
        train_nll = compute_mean_nll(train_logprobs, train_weights)

        # Log metrics
        metrics.update(
            num_sequences=len(batch),
            num_tokens=sum(d.model_input.length for d in batch),
            learning_rate=current_lr,
            train_mean_nll=train_nll,
            progress=step / n_train_batches,
            time_total=time.time() - start_time,
        )
        ml_logger.log_metrics(metrics=metrics, step=step)

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
    chz.nested_entrypoint(main)
