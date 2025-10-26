#!/usr/bin/env python3
"""
Quick training test without GUI (5 episodes only)
"""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Run without display

from toh import TowerOfHanoiEnv
from dqn_agent import DQNAgent
from util import flatten_state
import numpy as np

print("Testing basic training loop (no GUI)...")

# Setup
env = TowerOfHanoiEnv(num_discs=3)
state_size = np.prod(env.observation_space.shape)
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

print(f"Environment: {env.num_discs} discs")
print(f"State size: {state_size}, Action size: {action_size}")
print("\nRunning 5 training episodes...")

for episode in range(1, 6):
    state = env._reset()
    total_reward = 0
    step = 0
    success = False
    
    while step < 100:
        flat_state = flatten_state(state, env.num_discs)
        flat_state = np.reshape(flat_state, [1, agent.state_size])
        action = agent.act(flat_state)
        
        next_state, reward, done, _ = env.step(action)
        
        next_flat_state = flatten_state(next_state, env.num_discs)
        next_flat_state = np.reshape(next_flat_state, [1, agent.state_size])
        agent.remember(flat_state, action, reward, next_flat_state, done)
        
        state = next_state
        total_reward += reward
        step += 1
        
        if done:
            success = len(env.state[2]) == env.num_discs
            break
    
    # Train
    if len(agent.memory) > 16:
        agent.replay()
    
    status = "✓" if success else "✗"
    print(f"Episode {episode}: {status} | Steps: {step} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f}")

print("\n✅ Training loop test completed successfully!")
print("The full GUI training should work fine.")
