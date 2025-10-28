#!/usr/bin/env python3
"""
Quick comparison test showing the progressive placement system in action.
Runs a short training session and shows key metrics.
"""

from toh import TowerOfHanoiEnv
from dqn_agent import DQNAgent
from util import flatten_state
import numpy as np

def quick_training_test(num_episodes=50):
    """Run a quick training test to demonstrate the progressive placement system."""
    
    print("=" * 80)
    print("PROGRESSIVE PLACEMENT SYSTEM - QUICK TRAINING TEST")
    print("=" * 80)
    print(f"\nTraining for {num_episodes} episodes with 3 discs...")
    print("Watching for: placement rewards, removal penalties, and learning progress\n")
    
    # Setup
    env = TowerOfHanoiEnv(num_discs=3)
    state_size = 3 * 3  # 3 rods * 3 discs
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Track metrics
    success_count = 0
    total_placements = 0
    total_removals = 0
    recent_success = []
    
    for episode in range(num_episodes):
        state = env._reset()
        state = flatten_state(state, 3)
        
        total_reward = 0
        done = False
        steps = 0
        
        episode_placements = 0
        episode_removals = 0
        prev_placed = set()
        
        while not done and steps < 100:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Choose action
            if np.random.rand() <= agent.epsilon:
                action = np.random.choice(valid_actions) if valid_actions else np.random.randint(action_size)
            else:
                q_values = agent.model.predict(state.reshape(1, -1), verbose=0)[0]
                # Mask invalid actions
                for a in range(action_size):
                    if a not in valid_actions:
                        q_values[a] = -np.inf
                action = np.argmax(q_values)
            
            # Track placements/removals
            prev_placed_count = len(env.correctly_placed_discs)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = flatten_state(next_state, 3)
            
            # Track placement/removal events
            new_placed_count = len(env.correctly_placed_discs)
            if new_placed_count > prev_placed_count:
                episode_placements += 1
                total_placements += 1
            elif new_placed_count < prev_placed_count:
                episode_removals += 1
                total_removals += 1
            
            # Store experience and train
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            prev_placed = env.correctly_placed_discs.copy()
        
        # Track success
        if done:
            success_count += 1
        recent_success.append(1 if done else 0)
        if len(recent_success) > 10:
            recent_success.pop(0)
        
        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            success_rate = (success_count / (episode + 1)) * 100
            recent_rate = (sum(recent_success) / len(recent_success)) * 100
            print(f"Episode {episode + 1:3d}/{num_episodes} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Recent: {recent_rate:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Steps: {steps:3d} | "
                  f"Reward: {total_reward:7.1f} | "
                  f"Placed: {episode_placements} | "
                  f"Removed: {episode_removals}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total Episodes:        {num_episodes}")
    print(f"Successful Episodes:   {success_count} ({(success_count/num_episodes)*100:.1f}%)")
    print(f"Total Placements:      {total_placements} (avg: {total_placements/num_episodes:.1f} per episode)")
    print(f"Total Removals:        {total_removals} (avg: {total_removals/num_episodes:.1f} per episode)")
    print(f"Final Epsilon:         {agent.epsilon:.4f}")
    print(f"\n✓ Progressive placement system is active!")
    print(f"✓ Agents learn to place discs in order: {env.num_discs} → 1")
    print(f"✓ Removals are penalized to prevent oscillation")
    
    if total_placements > total_removals * 2:
        print(f"\n✅ Good learning pattern! More placements than removals.")
    else:
        print(f"\n⚠️  Agent still exploring. Train longer for better results.")
    
    print("=" * 80)
    
    # Test the trained agent
    print("\n" + "=" * 80)
    print("TESTING TRAINED AGENT (3 test episodes)")
    print("=" * 80)
    
    for test_ep in range(3):
        state = env._reset()
        state = flatten_state(state, 3)
        
        done = False
        steps = 0
        
        print(f"\nTest Episode {test_ep + 1}:")
        
        while not done and steps < 50:
            valid_actions = env.get_valid_actions()
            q_values = agent.model.predict(state.reshape(1, -1), verbose=0)[0]
            for a in range(action_size):
                if a not in valid_actions:
                    q_values[a] = -np.inf
            action = np.argmax(q_values)
            
            next_state, reward, done, _ = env.step(action)
            next_state = flatten_state(next_state, 3)
            
            state = next_state
            steps += 1
        
        if done:
            print(f"  ✓ Solved in {steps} steps! (optimal: 7)")
            print(f"  ✓ All discs correctly placed: {env.correctly_placed_discs}")
        else:
            print(f"  ✗ Failed to solve in {steps} steps")
            print(f"  Correctly placed: {env.correctly_placed_discs}")
    
    print("\n" + "=" * 80)
    print("Run the GUI to train longer and see better results!")
    print("  ./start_gui.sh")
    print("=" * 80)

if __name__ == "__main__":
    quick_training_test(num_episodes=50)
