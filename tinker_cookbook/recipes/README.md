# Cookbook Recipes

## Ready-to-go Post-training Examples

Navigate to each subfolder, and you check out the commands and expected performance.

Here are some helpful cli arguments that all of our examples should support:
- `wandb_project`: When provided, logs will be sent to your Weights & Biases project. Without this argument, training scripts save logs locally only
- `log_path`: Controls where training artifacts are saved
  - Default behavior: If not specified, each run generates a unique name and saves to `/tmp/tinker-examples`
  - Output files:
    - `{log_path}/metrics.jsonl` saves trainig metrics.
    - `{log_path}/checkpoints.jsonl` records all the checkpoints saved during training. You can share these checkpoints for model release, offline evaluation, etc.
  - Resuming: When using an existing log_path, you can either overwrite the previous run or resume training. This is particularly useful for recovering from runtime interruptions

## Starting from Scratch

Launch Script Template
- `rl_basic.py`: Template for reinforcement learning configurations
- `sl_basic.py`: Template for supervised learning configurations

Minimal training loops
- `rl_loop.py`: Minimal reinforcement learning training loop
- `sl_loop.py`: Minimal supervised learning training loop
