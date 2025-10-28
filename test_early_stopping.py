#!/usr/bin/env python3
"""
Test early stopping and performance tracking.
"""

from toh import TowerOfHanoiEnv
from dqn_agent import DQNAgent
from util import flatten_state
import numpy as np

def test_early_stopping():
    """Test the early stopping logic."""
    
    print("=" * 80)
    print("TESTING EARLY STOPPING MECHANISM")
    print("=" * 80)
    print("\nSimulating training with performance degradation...")
    print("Early stopping should trigger when moving average stops improving.\n")
    
    # Setup
    env = TowerOfHanoiEnv(num_discs=3)
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Simulate parameters
    patience = 20  # Stop after 20 episodes without improvement
    min_improvement = 0.05  # Need 5% improvement
    best_avg_steps = float('inf')
    episodes_without_improvement = 0
    
    episode_steps_list = []
    
    # Simulate 100 episodes with initial improvement then plateau
    print("Episode | Steps | Moving Avg | Best Avg | No Improve Count | Status")
    print("-" * 80)
    
    for episode in range(1, 101):
        # Simulate steps (improving initially, then plateauing/worsening)
        if episode < 30:
            # Initial learning - steps decrease
            steps = 50 - episode
        elif episode < 60:
            # Good performance - around 20-25 steps
            steps = np.random.randint(18, 27)
        else:
            # Performance degrades - steps increase
            steps = 20 + (episode - 60) * 0.5 + np.random.randint(-3, 3)
        
        episode_steps_list.append(steps)
        
        # Calculate moving average
        window_size = min(20, episode)
        recent_steps = episode_steps_list[-window_size:]
        moving_avg_steps = sum(recent_steps) / len(recent_steps)
        
        # Early stopping check (after initial exploration)
        status = "learning"
        if episode >= 30:  # Allow initial learning phase
            if moving_avg_steps < best_avg_steps * (1 - min_improvement):
                # Significant improvement!
                best_avg_steps = moving_avg_steps
                episodes_without_improvement = 0
                status = "✓ IMPROVED"
            else:
                episodes_without_improvement += 1
                status = f"no improve ({episodes_without_improvement}/{patience})"
                
                # Check if we should stop
                if episodes_without_improvement >= patience:
                    print(f"{episode:3d}     | {steps:5.1f} | {moving_avg_steps:10.2f} | {best_avg_steps:8.2f} | {episodes_without_improvement:16d} | {status}")
                    print("\n" + "=" * 80)
                    print(f"🛑 EARLY STOPPING TRIGGERED AT EPISODE {episode}")
                    print("=" * 80)
                    print(f"No improvement for {patience} consecutive episodes")
                    print(f"Best moving average achieved: {best_avg_steps:.2f} steps")
                    print(f"Current moving average: {moving_avg_steps:.2f} steps")
                    print("\nThis prevents wasting time on non-improving training!")
                    print("=" * 80)
                    break
        
        # Print every 5 episodes
        if episode % 5 == 0 or episode < 10:
            print(f"{episode:3d}     | {steps:5.1f} | {moving_avg_steps:10.2f} | {best_avg_steps:8.2f} | {episodes_without_improvement:16d} | {status}")
    
    print("\n" + "=" * 80)
    print("EARLY STOPPING BENEFITS")
    print("=" * 80)
    print("✓ Stops training when performance plateaus or degrades")
    print("✓ Saves computational time and resources")
    print("✓ Prevents overfitting and performance degradation")
    print("✓ Automatically identifies optimal training duration")
    print("=" * 80)

if __name__ == "__main__":
    test_early_stopping()
