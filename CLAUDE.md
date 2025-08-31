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

## Current Status & Issues Fixed (Dec 31, 2024)

### Performance Issues Identified
The IQN implementation was performing poorly (scores ~13-24 on CartPole instead of 200+). The following bugs were identified and fixed:

### Bugs Fixed
1. **Soft vs Hard Updates**: Changed from soft updates back to hard updates was causing instability. Reverted to use soft updates with TAU=1e-3
2. **Batching Issue**: `agent.act()` wasn't handling single states properly - needed to add batch dimension with `unsqueeze(0)`
3. **Action Type Issue**: `random.choices()` returns lists, needed conversion to numpy arrays
4. **Multi-worker Handling**: Fixed batching logic to handle both single and multiple worker scenarios
5. **SubprocVecEnv Hanging**: Single worker mode was trying to use subprocess wrapper unnecessarily

### Recommended Hyperparameters for CartPole
```bash
python run.py -env CartPole-v1 -info iqn_fixed -frames 50000 -eval_every 5000 -N 8 -lr 2.5e-4 -bs 32 -w 1
```
- Use `-w 1` (single worker) to avoid subprocess issues
- Use `-N 8` instead of 32 for faster computation
- Use `-lr 2.5e-4` (much higher than the 5e-5 that was causing poor performance)
- Use `-bs 32` for reasonable batch size

### Known Issues
- MultiPro.SubprocVecEnv hangs when creating environments
- Workaround: Use `-w 1` for single worker mode
- Alternative: Use `test_run.py` which bypasses MultiPro entirely

### Files Modified
- `agent.py`: Fixed batching, action types, reverted to soft updates
- `model.py`: Fixed pis initialization (though reverted)
- `run.py`: Added single worker wrapper, debug outputs

### Expected Performance
With these fixes, the agent should reach 200+ scores (solving CartPole) within 10,000-20,000 frames.

## Commands

### Running Training

Basic training on CartPole (RECOMMENDED):
```bash
python run.py -env CartPole-v1 -info iqn_run1 -frames 50000 -eval_every 5000 -N 8 -lr 2.5e-4 -bs 32 -w 1
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
- `-lr`: Learning rate (default: 2.5e-4) **CRITICAL: Don't use values below 1e-4**
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