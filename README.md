# FPO++ IsaacLab Experiments (Reproducibility Study)

This repository contains the IsaacLab experiments for our Deep Generative Modeling group project, where we reproduce and analyze the FPO++ algorithm. FPO++ replaces the traditional Gaussian action distributions used in Proximal Policy Optimization (PPO) with a learned continuous normalizing flow model that maps noise to actions via an ODE. 

This repository integrates our implementation of FPO++ as an [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/) extension to evaluate velocity-conditioned locomotion for 2 robots (Go2 and G1).

## Setup

**Prerequisites**: Linux, NVIDIA GPU with CUDA 12.1+.

```bash
# Install everything (conda env, IsaacSim 4.5, IsaacLab, isaaclab_fpo).
bash setup_env.sh

# Activate the environment (run this at the start of every session).
source source_env.sh
```

This creates a conda environment named `isaaclab_fpo`.

---

## Training

All FPO++ training is launched via `isaaclab_fpo/scripts/train.py`. Per-task hyperparameters are defined in `isaaclab_fpo/isaaclab_fpo/task_cfgs.py`.

### Velocity-Conditioned Locomotion

```bash
# Unitree Go2.
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless

# Unitree G1 (humanoid).
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-G1-v0 --headless
```
### Expected Results & Baselines

**Expected returns** (4096 envs):

| Robot | Iterations | Final Return |
|-------|-----------|--------------|
| Go2   | 1500      | ~40          |
| G1    | 2000      | ~37          |

---

## Experimental Configurations: Ratio Modes & Trust Regions

As part of our study into the generative policy mechanics of FPO++, we have implemented and tested various clipping constraints and annealing schedules. You can override these hyperparameters via positional arguments during training.

### Ratio Modes
Controls how the algorithm transitions between different probability ratio constraints throughout training.

| Mode | Description | Command Line Override |
| :--- | :--- | :--- |
| **`per_action`** | Standard PPO with hard clipping constraint. | `agent.algorithm.ratio_mode=per_action` |
| **`per_sample`** | Structured Policy Optimization with quadratic penalty. | `agent.algorithm.ratio_mode=per_sample` |
| **`hybrid1`** | First-half cosine annealing from per-action to per-sample, second-half per-sample. | `agent.algorithm.ratio_mode=hybrid1` |
| **`hybrid2`** | Linear annealing from per-action to per-sample. | `agent.algorithm.ratio_mode=hybrid2` |
| **`hybrid3`** | First-half strictly per-action, second-half strictly per-sample. | `agent.algorithm.ratio_mode=hybrid3` |

### Trust Region Modes
Dictates the specific trust region penalty or clipping mechanism applied to the policy updates.

| Mode | Description | Command Line Override |
| :--- | :--- | :--- |
| **`ppo`** | Standard PPO with hard clipping constraint. | `agent.algorithm.trust_region_mode=ppo` |
| **`spo`** | Structured Policy Optimization. Uses a smoother quadratic penalty instead of hard clipping: `loss = -mean(ratio * adv - abs(adv) / (2*eps) * (ratio - 1)^2)` | `agent.algorithm.trust_region_mode=spo` |
| **`aspo`** | Asymmetric SPO (Default). Uses standard PPO clipping for positive advantages, and SPO for negative advantages. | `agent.algorithm.trust_region_mode=aspo` |

**Example Command (Sweep-friendly):**
```bash
python isaaclab_fpo/scripts/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless \
    agent.algorithm.ratio_mode=hybrid1 \
    agent.algorithm.trust_region_mode=aspo \
    agent.algorithm.learning_rate=3e-4 \
    agent.algorithm.n_samples_per_action=32
```


### Common Logging & Environment Options

| Flag                 | Description                                      |
| -------------------- | ------------------------------------------------ |
| `--logger wandb`     | Enable W&B logging (default: tensorboard)        |
| `--log_project_name` | W&B project name (default: `isaaclab`)           |
| `--run_name`         | Suffix appended to the timestamped run directory |
| `--num_envs`         | Number of parallel environments                  |
| `--max_iterations`   | Override max training iterations                 |
| `--seed`             | Random seed (`-1` for random)                    |

---

## Playback (Viser)

Visualize a trained policy in the browser using [Viser](https://viser.studio/):

```bash
# Load checkpoint from a local path.
python isaaclab_fpo/scripts/play_with_viser.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --checkpoint logs/isaaclab_fpo/unitree_go2_flat_flow/2025-01-01_00-00-00/model_1500.pt \
    --headless --viser --num_envs 1

# Load checkpoint from W&B (entity/project/run_id from the run's URL).
python isaaclab_fpo/scripts/play_with_viser.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --wandb-run-path my-entity/my-project/abc123xy \
    --wandb-checkpoint model_1500.pt \
    --headless --viser --num_envs 4
```


Then open `http://localhost:8080` in your browser. 
*(Note: The `--wandb-run-path` is the `entity/project/run_id` from your W&B run URL)*.

---

## Directory Layout

text
isaaclab_experiments/
├── isaaclab_fpo/                         # FPO++ package (algorithm + IsaacLab integration)
│   ├── scripts/
│   │   ├── train.py                     # Main training entry point
│   │   ├── play_with_viser.py           # Viser-based policy playback (browser)
│   │   ├── play.py                      # IsaacSim viewer playback
│   │   ├── play_plot.py                 # Playback with live reward plotting
│   ├── viser_assets/                    # Pre-extracted robot meshes for Viser
│   └── isaaclab_fpo/
│       ├── task_cfgs.py                 # Per-task FPO++ hyperparameters (TASK_CONFIGS registry)
│       ├── rl_cfg.py                    # FPO++ config dataclasses
│       ├── algorithms/fpo.py            # FPO++ algorithm (flow-based PPO)
│       ├── modules/actor_critic.py      # Flow actor + value critic networks
│       ├── runners/on_policy_runner.py  # Training loop with EMA, eval, multi-GPU
│       ├── storage/rollout_storage.py   # Rollout buffer with CFM loss storage
│       ├── wrapper.py                   # VecEnv wrapper for IsaacLab
│       ├── cli_args.py                  # CLI argument helpers
│       └── patches.py                   # IsaacLab monkey-patches for sweep support
│
├── thirdparty/
│   ├── IsaacLab/                        # NVIDIA Isaac Lab (git submodule)
│   │   └── source/
│   │       ├── isaaclab/                # Core framework
│   │       ├── isaaclab_tasks/          # Task definitions (locomotion, etc.)
│   │       └── isaaclab_assets/         # Robot USD assets & configs
│
├── expected_training_curves.png          # Reference training curves
├── expected_eval_curves.png              # Reference eval curves
├── setup_env.sh                          # One-time environment setup
└── source_env.sh                         # Activate conda env


---

## Acknowledgements

This reproducibility study adapts work from the original authors of FPO++. Thank you to the **Amazon FAR** team for their foundational implementation:
* **[amazon-far/fpo-control](https://github.com/amazon-far/fpo-control/tree/main)** (Original FPO++ Implementation)

Additionally, the `isaaclab_fpo` package combines and adapts code from the following foundational robotics and RL projects:

| Source | License | What we adapted |
|--------|---------|-----------------|
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (ETH Zurich + NVIDIA) | BSD-3-Clause | Actor-critic, on-policy runner, rollout storage, normalizer, logging utilities |
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) (NVIDIA) | BSD-3-Clause | VecEnv wrapper, config dataclasses, training/play/evaluate scripts, ONNX exporter |

`IsaacLab/` is included as a git submodule under its original license. Original copyright headers are retained in all adapted files across the codebase.
