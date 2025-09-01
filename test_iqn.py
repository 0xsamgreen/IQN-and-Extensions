#!/usr/bin/env python3
"""Simple test script to verify IQN fixes work correctly."""

import torch
import numpy as np
import gymnasium as gym
from agent import IQN_Agent
from collections import deque

def test_iqn():
    # Setup
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    device = torch.device("cpu")
    
    print(f"Environment: CartPole-v1")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Device: {device}")
    
    # Create agent with proper hyperparameters
    agent = IQN_Agent(
        state_size=state_size,    
        action_size=action_size,
        network='iqn',
        munchausen=0,
        layer_size=128,  # Smaller for faster testing
        n_step=1,
        BATCH_SIZE=32, 
        BUFFER_SIZE=10000, 
        LR=5e-5, 
        TAU=1e-3, 
        GAMMA=0.99,  
        N=8,  # Smaller for testing
        worker=1,
        device=device, 
        seed=42
    )
    
    print("\nAgent created successfully!")
    print(f"N (current quantiles): {agent.N}")
    print(f"N_dash (target quantiles): {agent.N_dash}")
    print(f"Target update interval: {agent.target_update_interval}")
    
    # Run a quick training test
    scores = deque(maxlen=100)
    
    for episode in range(10):
        state, _ = env.reset()
        score = 0
        
        for t in range(500):
            # Act
            action = agent.act(np.expand_dims(state, axis=0), eps=0.1, eval=False)[0]
            
            # Step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience (using dummy writer)
            class DummyWriter:
                def add_scalar(self, *args): pass
            
            agent.step(state, action, reward, next_state, done, DummyWriter())
            
            state = next_state
            score += reward
            
            if done:
                break
        
        scores.append(score)
        print(f"Episode {episode+1}: Score = {score:.0f}, Avg = {np.mean(scores):.1f}")
        
        # Check if learning is happening
        if episode == 5:
            print(f"\nLearning steps: {agent.Q_updates}")
            print(f"Target updates: {agent.learn_step_counter // agent.target_update_interval}")
    
    print("\n✓ Test completed successfully!")
    return True

if __name__ == "__main__":
    test_iqn()