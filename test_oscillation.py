#!/usr/bin/env python3
"""Test oscillation detection in DQN Agent."""
import numpy as np
from dqn_agent import DQNAgent
from toh import TowerOfHanoiEnv
from util import flatten_state

def test_oscillation_detection():
    """Test that oscillation detection catches repetitive actions."""
    print("="*80)
    print("Testing Oscillation Detection")
    print("="*80 + "\n")
    
    num_discs = 3
    env = TowerOfHanoiEnv(num_discs=num_discs)
    state_size = num_discs * 3
    action_size = 6
    
    agent = DQNAgent(state_size, action_size, architecture_name="Small (24-24)")
    agent.epsilon = 0.0  # No random exploration - only exploit to test deterministic behavior
    
    # Start episode
    agent.reset_episode()
    state = env._reset()
    
    print("Initial state:", env.state)
    print(f"Agent epsilon: {agent.epsilon}")
    print(f"Oscillation threshold: {agent.oscillation_threshold}")
    print("\nSimulating episode with potential for oscillation...\n")
    
    for step in range(30):
        flat_state = flatten_state(state, num_discs)
        flat_state = np.reshape(flat_state, [1, state_size])
        
        valid_actions = env.get_valid_actions()
        action = agent.act(flat_state, valid_actions)
        
        from_rod, to_rod = env.decode_action(action)
        disc = env.state[from_rod][-1] if env.state[from_rod] else None
        
        next_state, reward, done, _ = env.step(action)
        
        print(f"Step {step:2d}: Action {action} ({from_rod}→{to_rod}), Disc {disc}, Reward {reward:6.1f}, Valid actions: {valid_actions}")
        print(f"         Recent actions: {list(agent.recent_actions)}")
        print(f"         Blocked oscillations: {agent.blocked_oscillations}")
        print(f"         State: {env.state}\n")
        
        if done:
            print(f"✓ Puzzle solved in {step+1} steps!")
            break
        
        state = next_state
        
        # Stop if we detect we're stuck (same action repeated many times)
        if len(agent.recent_actions) >= 10:
            last_10 = list(agent.recent_actions)[-10:]
            if len(set(last_10)) == 1:
                print("❌ ERROR: Agent stuck repeating same action 10 times!")
                print(f"   Blocked oscillations: {agent.blocked_oscillations}")
                break
    
    print("\n" + "="*80)
    print("Test complete")
    print("="*80)

if __name__ == "__main__":
    test_oscillation_detection()
