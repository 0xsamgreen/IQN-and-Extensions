import gymnasium as gym
import numpy as np
from agent import IQN_Agent

# Test basic agent initialization
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape  # Use full shape tuple, not just the first dimension
action_size = env.action_space.n

print(f"State size: {state_size}, Action size: {action_size}")

# Try to create agent
try:
    agent = IQN_Agent(state_size, action_size, network="iqn", seed=0, device="cpu",
                      munchausen=0, layer_size=512, n_step=1, 
                      BATCH_SIZE=8, BUFFER_SIZE=int(1e5), LR=2.5e-4,
                      TAU=1e-3, GAMMA=0.99, N=8, worker=1)
    print("Agent created successfully!")
    
    # Test getting an action
    state, _ = env.reset()
    action = agent.act(state, eps=0.1)
    print(f"Action: {action}")
    
    # Convert action to scalar if it's an array
    if isinstance(action, np.ndarray):
        action = action[0]
    
    # Try a step without writer (dummy)
    class DummyWriter:
        def add_scalar(self, *args, **kwargs):
            pass
    
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done, DummyWriter())
    print("Step completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()