# ReLIC: Reinforcement Learning for Interlimb Coordination

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)


This repository contains the official implementation for the paper "Versatile Loco-Manipulation through Flexible Interlimb Coordination", accepted to CoRL 2025.

[**Project Website**](https://relic-locoman.rai-inst.com/) | [**arXiv Paper**](https://arxiv.org/abs/2506.07876) | [**Blog Post**](https://rai-inst.com/resources/blog/reinforcement-learning-for-flexible-loco-manipulation/) | [**X**](https://x.com/rai_inst/status/1971590845817045218) | [**Threads**](https://www.threads.com/@robotics_and_ai_institute/post/DPEdj0UFEnm?xmt=AQF0fkVTtJFB-UTiEjOQATobby44KSl_DdfDymPMpr42Vg)

---
*Our ReLIC policy enables a quadruped robot to walk with three legs and manipulate with the arm and one leg.*

## Overview

**Reinforcement Learning for Interlimb Coordination (ReLIC)** is an approach that enables versatile loco-manipulation through flexible interlimb coordination. The core of our method is a single, adaptive controller that learns to seamlessly coordinate all of a robot's limbs. It intelligently bridges the execution of precise manipulation motions with the generation of stable, dynamic gaits, allowing the robot to interact with its environment in a versatile and robust manner.

## Install
This project uses [Pixi](https://pixi.sh/latest/installation/) to manage dependencies and ensure a consistent development environment. You won't need to use `conda` or `virtualenv` separately or manually install `IsaacSim` or `IsaacLab`. Just follow the steps below to get started.

1. **Install Pixi**
    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

2. **Clone the repository**
    ```bash
    git clone https://github.com/bdaiinstitute/relic.git && cd relic
    ```

3. **Install dependencies**
    ```bash
    pixi install
    ```

4. **Activate the environment**
    ```bash
    pixi shell
    ```

Alternatively, you can install the project without Pixi by following the standard installation guides for [IsaacLab](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/isaaclab_pip_installation.html) and its [extensions](https://github.com/isaac-sim/IsaacLabExtensionTemplate/tree/main?tab=readme-ov-file#installation).

## Training and Eval

```bash
python scripts/rsl_rl/train.py --task Isaac-Spot-Interlimb-Phase-1-v0 --headless
python scripts/rsl_rl/play.py --task Isaac-Spot-Interlimb-Play-v0 --center
```

To achieve optimal deployment results, we implemented a weight curriculum with multiple training phases. Users can fine-tune the models from `Phase-2` to `Phase-4` to reproduce the results presented in our paper. The pre-trained weights can be found in `relic/source/relic/relic/assets/spot/pretrained`.

## Deployment

Please find deployment code with Boston Dynamics's Spot robot at our deployment branch.

## Adapted Code

We use RSL_RL for RL training and adapt the following scripts from [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate/tree/main?tab=readme-ov-file#installation)
- `scripts/rsl_rl/*`
- `source/relic/pyproject.toml`
- `source/relic/setup.py`
