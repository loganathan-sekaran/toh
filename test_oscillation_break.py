#!/usr/bin/env python3
"""
Test script to verify oscillation detection and forced exploration.
This simulates repeated actions to trigger oscillation breaking mechanism.
"""

import numpy as np
from dqn_agent import DQNAgent

def test_oscillation_breaking():
    """Test that oscillation detection forces exploration."""
    print("=" * 60)
    print("Testing Oscillation Detection and Forced Exploration")
    print("=" * 60)
    
    # Create agent
    agent = DQNAgent(state_size=27, action_size=6, architecture_name="Small (24-24)")
    
    # Simulate oscillating behavior (repeating same 2 actions)
    dummy_state = np.zeros((1, 27))
    
    print("\n1. Simulating 2-action oscillation (should trigger after 6 steps):")
    print("-" * 60)
    
    # Manually add alternating actions to trigger detection
    for i in range(10):
        # Alternate between action 0 and 1
        agent.recent_actions.append(0 if i % 2 == 0 else 1)
        
        if i >= 5:  # After 6 actions, oscillation should be detected
            print(f"Step {i+1}: recent_actions = {list(agent.recent_actions)[-6:]}")
            
            if agent.force_random_actions > 0:
                print(f"  ✓ FORCED EXPLORATION ACTIVE: {agent.force_random_actions} random actions remaining")
                print(f"  ✓ Epsilon boosted to: {agent.epsilon:.3f}")
                break
    
    # Test forced random counter
    print(f"\nInitial force_random_actions counter: {agent.force_random_actions}")
    
    print("\n2. Testing forced random actions:")
    print("-" * 60)
    for i in range(5):
        action = agent.act(dummy_state, valid_actions=[0,1,2,3,4,5])
        print(f"Action {i+1}: {action}, force_random remaining: {agent.force_random_actions}")
    
    print("\n3. Testing repetitive single action (AAAA...):")
    print("-" * 60)
    agent2 = DQNAgent(state_size=27, action_size=6, architecture_name="Small (24-24)")
    
    # Add same action 6 times
    for i in range(6):
        agent2.recent_actions.append(3)
    
    print(f"Recent actions: {list(agent2.recent_actions)}")
    action = agent2.act(dummy_state, valid_actions=[0,1,2,3,4,5])
    print(f"Forced exploration active: {agent2.force_random_actions > 0}")
    print(f"Epsilon after detection: {agent2.epsilon:.3f}")
    
    print("\n" + "=" * 60)
    print("✓ Oscillation Breaking Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_oscillation_breaking()
