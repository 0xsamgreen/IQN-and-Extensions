# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**IMPORTANT: Always activate the virtual environment before running any code:**

```bash
source venv/bin/activate
```

## Project Overview

This is a PyTorch implementation of Implicit Quantile Networks (IQN) for Distributional Reinforcement Learning with extensions including:
- Prioritized Experience Replay (PER)
- Noisy Networks
- N-step bootstrapping
- Munchausen RL
- Dueling architecture
- Parallel environment training

## Commands

### Running Training

Basic training on CartPole:
```bash
python run.py -info iqn_run1
```

Training on Atari games (e.g., Pong):
```bash
python run.py -env PongNoFrameskip-v4 -info iqn_pong1
```

### Key Command Parameters

- `-agent`: Choose agent type (iqn, iqn+per, noisy_iqn, dueling, etc.)
- `-env`: Environment name (default: BreakoutNoFrameskip-v4)
- `-frames`: Training frames (default: 10M)
- `-N`: Number of quantiles (default: 8)
- `-w`: Number of parallel workers (default: 1)
- `-munchausen`: Enable Munchausen RL (0 or 1)

### Monitoring Training

View training progress:
```bash
tensorboard --logdir=runs
```

## Architecture

### Core Components

**IQN_Agent** (`agent.py`): Main agent class that handles:
- Action selection with epsilon-greedy or noisy exploration
- Experience replay buffer management (standard or prioritized)
- Network updates with IQN loss calculation
- Soft target network updates

**IQN Model** (`model.py`): Neural network architectures:
- Base IQN with quantile sampling
- Dueling IQN variant
- Noisy linear layers for exploration
- Cosine embedding for quantile representation

**Replay Buffers** (`ReplayBuffers.py`):
- `ReplayBuffer`: Standard experience replay with n-step returns
- `PrioritizedReplay`: Prioritized experience replay with importance sampling

**Environment Wrappers** (`wrapper.py`):
- Atari-specific wrappers (FireReset, MaxAndSkip, ProcessFrame84, etc.)
- Frame stacking and preprocessing

**Parallel Training** (`MultiPro.py`):
- `SubprocVecEnv`: Manages multiple environment instances for parallel data collection

### Training Flow

1. **run.py** initializes environments (single or parallel), creates agent, and manages training loop
2. Agent collects experiences using epsilon-greedy or noisy exploration
3. Experiences stored in replay buffer (with n-step returns if configured)
4. Agent samples batches and computes IQN loss with sampled quantiles
5. Networks updated via gradient descent with optional Munchausen bonus
6. Target network soft-updated periodically

### Key Implementation Details

- Quantiles sampled uniformly from [0,1] and embedded using cosine basis functions
- Huber loss used for quantile regression
- Supports both image-based (Atari) and vector-based (CartPole) observations
- Parallel environments increase batch size proportionally to worker count