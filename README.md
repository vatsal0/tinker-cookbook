// TODO(tianyi) Fancy hero image?


Tinker cookbook collects recommended programming patterns, reusable utilities, and extensible abstractions to help people build on [Tinker](https://tinker-docs.thinkingmachines.ai/).

## Installation

1. Obtain a Tinker API token and export it as `TINKER_API_KEY`. // TODO(tianyi): add onboarding flow link
2. Install tinker python client via `pip install git+https://github.com/thinking-machines-lab/tinker.git` // TODO(tianyi): update to pypi
3. As a starting point, we recommend cloning this repo locally and installing it via `pip install -e .`.

## Usage

We build Tinker cookbook to allow flexible usage. You can run our examples, build your own training loop, or simply import useful utilities from this repo.

### Running our examples

`tinker_cookbook/supervised/train.py` and `tinker_cookbook/rl/train.py` contain our reference entrypoints for supervised learning and reinforcement learning accordingly.

Navigate to `tinker_cookbook/recipes` and you will find ready-to-go post-training examples. Here are the list of examples you can try out:
- `chat_sl` shows supervised fine-tuning on Tulu3
- `prompt_distillation` XXXX // TODO(tianyi): add take away message
- `math_rl` demontrates Refinforcement Learning with Verifiable Reward (RLVR) on math problems
- `multiplayer_rl` leverages the flexibility of Tinker to learn on multiplayer / multi-model games
- `tool_use/search` replicates a recent academic paper on using RL to teach the ability to use a vector search tool.

### Building your own

`sl_basic.py` and `rl_basic.py` remove most of our abstractions and provide clean starting points for building your own projects.

### Import our utilities

Tinker cookbook includes several patterns we like. Here's a quick overview,
- [renderers]() converts tokens from/to structured chat message objects
- [hyperparam_utils]() helps calculate hyperparameters suitable for LoRAs
- [evaluation]() shows how to evaluate Tinker models and also integrate with InspectAI to make evaluating on standard benchmarks easy.

## Contributing

We welcome community contributions to Tinker cookbook. At the same time, we want to keep this official repo lean and hackable.
If you build a cool project, please share with us and we'd love to highlight them in `FEATURED_PROJECTS.md`.
If you want to help us improve the core utilities in Tinker cookbook, please be familiar with `CONTRIBUTING.md`. We also post ideas on where we could use help.

For general feedback, you can XXXX // TODO(tianyi): check with clare
