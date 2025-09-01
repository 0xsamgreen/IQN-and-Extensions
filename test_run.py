import gymnasium as gym
import numpy as np
import torch
from agent import IQN_Agent
import MultiPro

# Simple test without multiple workers
print("Creating environment...")
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape
action_size = env.action_space.n

print("Creating agent...")
agent = IQN_Agent(state_size=state_size,
                  action_size=action_size,
                  network="iqn",
                  munchausen=0,
                  layer_size=512,
                  n_step=1,
                  BATCH_SIZE=8,
                  BUFFER_SIZE=int(1e5),
                  LR=2.5e-4,
                  TAU=1e-3,
                  GAMMA=0.99,
                  N=8,
                  worker=1,
                  device="cpu",
                  seed=0)

print("Starting training loop...")

class DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass

writer = DummyWriter()

state, _ = env.reset()
for i in range(100):
    if i % 10 == 0:
        print(f"Step {i}")
    
    action = agent.act(state, eps=0.1)
    if isinstance(action, np.ndarray):
        action = action[0]
    
    next_state, reward, done, truncated, _ = env.step(action)
    done = done or truncated
    agent.step(state, action, reward, next_state, done, writer)
    
    if done:
        state, _ = env.reset()
    else:
        state = next_state

print("Training loop completed successfully!")