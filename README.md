# RL-Env-Kit: An End-to-End Reinforcement Learning Environment

A production-style reinforcement learning (RL) environment, built from scratch: a custom [Gymnasium](https://gymnasium.farama.org/)-compatible environment, a verifiable reward function, baseline training, evaluation, reproducible packaging with Docker, and CI.

> Replace `RL-Env-Kit` and the example task below with your own project name and domain.

---

## Why this project

Companies training agents (including LLM agents) need well-designed environments: clear state/action spaces, reliable rewards, reproducibility, and proper evaluation. This repo demonstrates the full pipeline, not just a training script:

- **Environment design**: custom `gymnasium.Env` with documented spaces
- **Reward engineering**: verifiable, deterministic reward with shaping options
- **Training**: baseline agents (PPO / DQN) using Stable-Baselines3
- **Evaluation**: seeded, repeatable benchmarks with metrics and plots
- **Engineering**: tests, type hints, Docker, CI, experiment tracking

---

## Features

- Gymnasium API compliant (`reset`, `step`, `render`, `close`) and passes `check_env`
- Configurable difficulty via YAML configs
- Deterministic seeding for reproducible results
- Reward function with unit tests (sparse and shaped variants)
- Baseline agents: Random, Heuristic, PPO, DQN
- Evaluation harness with success rate, mean return, and episode length
- TensorBoard / Weights & Biases logging
- Dockerized for one-command setup
- GitHub Actions CI (lint + tests)

---

## Project structure

```
rl-env-kit/
├── src/
│   ├── envs/
│   │   ├── __init__.py          # registers the env with Gymnasium
│   │   ├── custom_env.py        # the environment
│   │   └── rewards.py           # reward functions
│   ├── agents/
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   └── sb3_agent.py         # PPO / DQN wrappers
│   ├── train.py                 # training entry point
│   ├── evaluate.py              # evaluation entry point
│   └── utils/                   # seeding, logging, plotting
├── configs/
│   ├── easy.yaml
│   ├── medium.yaml
│   └── hard.yaml
├── tests/
│   ├── test_env_api.py          # gymnasium check_env
│   ├── test_rewards.py
│   └── test_determinism.py
├── notebooks/
│   └── analysis.ipynb
├── results/                     # metrics, plots, saved models
├── Dockerfile
├── requirements.txt
├── Makefile
└── README.md
```

---

## Environment specification

| Item | Description |
|---|---|
| **Task** | *Describe the goal, e.g. "navigate a grid to deliver items while avoiding obstacles"* |
| **Observation space** | e.g. `Box(low=0, high=1, shape=(N,), dtype=float32)` |
| **Action space** | e.g. `Discrete(4)` |
| **Episode start** | Randomized from the given seed |
| **Termination** | Goal reached, failure state hit |
| **Truncation** | Max steps (configurable) |
| **Reward** | See below |

### Reward design

| Event | Reward |
|---|---|
| Task success | `+1.0` |
| Each step (time penalty) | `-0.01` |
| Invalid action | `-0.1` |
| Failure | `-1.0` |

The sparse variant gives only success/failure. The shaped variant adds progress-based rewards. Both are selectable in config.

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/<your-username>/rl-env-kit.git
cd rl-env-kit
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the environment

```python
import gymnasium as gym
import src.envs  # registers the env

env = gym.make("RLEnvKit-v0", config="configs/easy.yaml")
obs, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### 3. Train

```bash
python -m src.train --algo ppo --config configs/easy.yaml --timesteps 500000 --seed 0
```

### 4. Evaluate

```bash
python -m src.evaluate --model results/ppo_easy/model.zip --config configs/easy.yaml --episodes 100
```

### 5. Visualize training

```bash
tensorboard --logdir results/
```

### Docker

```bash
docker build -t rl-env-kit .
docker run --rm rl-env-kit python -m src.evaluate --agent random --episodes 20
```

---

## Results

*Fill this in with your own numbers after running experiments.*

| Agent | Difficulty | Success rate | Mean return | Mean episode length |
|---|---|---|---|---|
| Random | Easy | – | – | – |
| Heuristic | Easy | – | – | – |
| PPO | Easy | – | – | – |
| PPO | Hard | – | – | – |

Learning curve plots are saved in `results/plots/`.

---

## Testing

```bash
pytest -q
```

Tests cover:
- Gymnasium API compliance (`check_env`)
- Reward correctness for edge cases
- Determinism: same seed gives identical trajectories

---

## Roadmap

- [ ] Add multi-agent variant
- [ ] Add curriculum learning across difficulty levels
- [ ] Add action masking for invalid actions
- [ ] Add a text/tool-use variant for LLM agents with verifiable rewards
- [ ] Publish as a pip package

---

## Tech stack

Python 3.10+, Gymnasium, Stable-Baselines3, PyTorch, NumPy, Pytest, Docker, GitHub Actions, TensorBoard.
