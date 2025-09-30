# Supervised Learning

## SFT on NoRobots

```bash
python -m tinker_cookbook.recipes.chat_sl.train
    model_name=Qwen/Qwen3-8B-Base \
    dataset=no_robots \
    learning_rate=5e-4 \
    batch_size=64 \
    lora_rank=64 \
    eval_every=20 \
    save_every=20 \
    wandb_project=cookbook_sl
```

After 140 steps of training, we achieve `"test/nll": 1.7878098487854004`. This example should finish within 10 minutes.

## SFT on Tulu3 dataset.

```bash
python -m tinker_cookbook.recipes.chat_sl.train
    model_name=Qwen/Qwen3-8B-Base \
    dataset=tulu3 \
    learning_rate=5e-4 \
    batch_size=128 \
    lora_rank=64 \
    eval_every=500 \
    save_every=500 \
    wandb_project=cookbook_sl
```
After 1740 steps of training, we achieve a test loss around 0.50. Running with the example hyperparameters takes around ~6h. By increasing lora_rank and lowering batch_size, we can achieve better test nll with longer run time.

## Add your own dataset

The base classes in `tinker-cookbook/tinker_cookbook/supervised/data.py` support loading new data in the following way:
- `SupervisedDatasetFromHFDataset` loads dataset on huggingface hub with a postprocessing function
- `StreamingSupervisedDatasetFromHFDataset` works simiarly, but supports streaming
- `FromConversationFileBuilder` supports data loading from a JSONL file
