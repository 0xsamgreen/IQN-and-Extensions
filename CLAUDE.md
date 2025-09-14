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

## Current Status (Dec 31, 2024)

The IQN implementation is now **working correctly** after fixing critical algorithm bugs. The agent successfully learns CartPole-v1 and other environments.

### Critical Bugs That Were Fixed

1. **Incorrect TD Error Shape**: Fixed TD error calculation from `(batch, N, N)` to correct `(batch, N, N_dash)` for proper pairwise quantile comparison
2. **Missing Separate Tau Sampling**: Now correctly samples different tau values for current network (tau) and target network (tau_dash)  
3. **Wrong Quantile Loss**: Fixed the quantile regression loss calculation with proper dimension handling
4. **Loss Aggregation**: Changed from sum to mean over N_dash dimension for proper gradient scaling
5. **Epsilon Decay**: Default eps_frames was 1M which prevented exploitation - now uses appropriate values for each environment

### Recommended Hyperparameters

#### CartPole-v1 (Tested and Verified)
```bash
python run.py -env CartPole-v1 -info iqn_cartpole -frames 20000 -eval_every 5000 -N 8 -lr 2.5e-4 -bs 32 -eps_frames 5000 -w 1
```

#### Key Parameters Explained
- `-N 8`: Number of quantiles (8 is sufficient for CartPole, use 32-64 for Atari)
- `-lr 2.5e-4`: Learning rate (tested and works well)
- `-bs 32`: Batch size for stable updates
- `-eps_frames 5000`: Epsilon decay frames (CRITICAL for CartPole - must be much shorter than default 1M)
- `-w 1`: Single worker (avoids subprocess issues)

### Expected Performance

With the fixed implementation:
- **CartPole-v1**: Reaches 50+ score by 5000 frames, 100+ by 10000 frames, potentially 200+ by 20000 frames (VERIFIED)
- **Atari Games**: Not yet tested with current fixes. May require additional debugging for image-based observations

## Commands

### Running Training

Basic training on CartPole (TESTED & VERIFIED):
```bash
python run.py -env CartPole-v1 -info iqn_cartpole -frames 20000 -eval_every 5000 -N 8 -lr 2.5e-4 -bs 32 -eps_frames 5000 -w 1
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
- `-w`: Number of parallel workers (default: 1) **USE 1 TO AVOID HANGING**
- `-lr`: Learning rate (default: 2.5e-4) **Tested value: 2.5e-4 works well for CartPole**
- `-munchausen`: Enable Munchausen RL (0 or 1)

### Monitoring Training

View training progress:
```bash
tensorboard --logdir=runs
```

For remote viewing, see README.md section on "Remote TensorBoard Viewing"

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