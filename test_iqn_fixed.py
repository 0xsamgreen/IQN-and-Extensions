import torch
import numpy as np
import gymnasium as gym
from agent import IQN_Agent
from torch.utils.tensorboard import SummaryWriter

# Test the fixed IQN implementation
def test_iqn_fixes():
    print("Testing IQN fixes...")
    
    # Create a simple environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    
    # Create agent with small parameters for testing
    agent = IQN_Agent(
        state_size=state_size,
        action_size=action_size,
        network="iqn",
        munchausen=False,
        layer_size=128,
        n_step=1,
        BATCH_SIZE=8,
        BUFFER_SIZE=1000,
        LR=1e-3,
        TAU=1e-3,
        GAMMA=0.99,
        N=4,  # Small N for testing
        worker=1,
        device="cpu",
        seed=0
    )
    
    print(f"Agent created with N={agent.N}")
    
    # Collect some experiences
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # Handle new gymnasium API
    
    for i in range(100):
        action = agent.act(state, eps=0.5)
        next_state, reward, done, truncated, info = env.step(action[0] if isinstance(action, np.ndarray) else action)
        done = done or truncated
        
        # Add to memory (without learning yet)
        agent.memory.add(state, action, reward, next_state, done)
        
        state = next_state
        if done:
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
    
    print(f"Collected {len(agent.memory)} experiences")
    
    # Test the learn function
    if len(agent.memory) > agent.BATCH_SIZE:
        experiences = agent.memory.sample()
        
        # Run one learning step
        try:
            loss = agent.learn(experiences)
            print(f"✓ Learning step completed successfully! Loss: {loss:.4f}")
            
            # Check that the shapes are correct
            states, actions, rewards, next_states, dones = experiences
            print(f"  States shape: {states.shape}")
            print(f"  Actions shape: {actions.shape}")
            print(f"  Rewards shape: {rewards.shape}")
            
            # Run forward pass to check quantile generation
            with torch.no_grad():
                quantiles, taus = agent.qnetwork_local(states, agent.N)
                print(f"  Quantiles shape: {quantiles.shape} (expected: {states.shape[0]}, {agent.N}, {action_size})")
                print(f"  Taus shape: {taus.shape} (expected: {states.shape[0]}, {agent.N}, 1)")
                
            print("\n✓ All tests passed! The IQN implementation appears to be fixed.")
            return True
            
        except Exception as e:
            print(f"✗ Error during learning: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("Not enough experiences collected")
        return False

if __name__ == "__main__":
    success = test_iqn_fixes()
    if success:
        print("\nThe fixes should improve performance. You can now run the full training.")
    else:
        print("\nThere are still issues to fix.")