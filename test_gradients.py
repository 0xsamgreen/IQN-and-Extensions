"""Test if gradients are flowing properly through the IQN network"""
import torch
import numpy as np
from agent import IQN_Agent
from torch.utils.tensorboard import SummaryWriter

def test_gradient_flow():
    print("Testing gradient flow in IQN...")
    
    # Simple test setup
    state_size = (4,)
    action_size = 2
    
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
        N=8,  # Small N for testing
        worker=1,
        device="cpu",
        seed=0
    )
    
    # Add some random experiences
    for _ in range(20):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.random() > 0.9
        agent.memory.add(state, action, reward, next_state, done)
    
    # Get initial parameters
    initial_params = {name: param.clone() for name, param in agent.qnetwork_local.named_parameters()}
    
    # Do one learning step
    if len(agent.memory) > agent.BATCH_SIZE:
        experiences = agent.memory.sample()
        loss = agent.learn(experiences)
        
        # Check if parameters changed
        params_changed = False
        for name, param in agent.qnetwork_local.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-6):
                params_changed = True
                max_change = (param - initial_params[name]).abs().max().item()
                print(f"  {name}: max change = {max_change:.6f}")
        
        if params_changed:
            print("✓ Gradients are flowing, parameters updated")
            print(f"  Loss value: {loss:.4f}")
        else:
            print("✗ WARNING: No parameter updates detected!")
            
        # Check gradient magnitudes
        print("\nGradient magnitudes:")
        for name, param in agent.qnetwork_local.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: {grad_norm:.6f}")
    
    return True

if __name__ == "__main__":
    test_gradient_flow()