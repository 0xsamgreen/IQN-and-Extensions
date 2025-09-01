"""
Quick performance test to compare before and after the IQN fixes
"""
import subprocess
import time

print("=" * 60)
print("IQN Performance Test - Fixed Implementation")
print("=" * 60)
print("\nRunning CartPole-v1 with fixed IQN...")
print("Command: python run.py -env CartPole-v1 -info iqn_performance_test -frames 20000 -eval_every 5000 -N 8 -lr 2.5e-4 -bs 32 -w 1")
print("\nExpected behavior:")
print("- Should reach 200+ score within 10,000-20,000 frames")
print("- Current scores (before fix): ~20-23 average")
print("\nStarting training...")
print("-" * 60)

# Run the training
cmd = ["python", "run.py", "-env", "CartPole-v1", "-info", "iqn_performance_test", 
       "-frames", "20000", "-eval_every", "5000", "-N", "8", "-lr", "2.5e-4", "-bs", "32", "-w", "1"]

print("\nThis will take a few minutes to complete...")
print("Look for the Average100 Score to see improvement")
print("-" * 60)