"""
Main launcher for Tower of Hanoi RL Training with GUI Visualization.
"""
import argparse
import sys
import os
import time
import numpy as np
from PyQt6.QtWidgets import QApplication
from toh import TowerOfHanoiEnv
from dqn_agent import DQNAgent
from visualizer import TowerOfHanoiVisualizer, create_visualizer
from train_with_gui import train_with_visualization, ModelManager


def test_model(num_discs=3, model_path=None, animation_speed=0.5):
    """
    Test a trained model with visualization.
    
    Args:
        num_discs: Number of discs
        model_path: Path to trained model file
        animation_speed: Speed of animation
    """
    print("="*80)
    print("Tower of Hanoi - Model Testing")
    print("="*80)
    
    # Initialize environment
    env = TowerOfHanoiEnv(num_discs=num_discs)
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(state_size, action_size)
    agent.epsilon = 0.0  # No exploration during testing
    
    # Load model
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from: {model_path}")
    else:
        print("Warning: No model loaded. Using random agent.")
    
    # Initialize visualizer
    visualizer = TowerOfHanoiVisualizer(num_discs=num_discs, animation_speed=animation_speed)
    
    # Test for multiple episodes
    num_test_episodes = 5
    successes = 0
    total_steps = 0
    
    print(f"\nTesting for {num_test_episodes} episodes...\n")
    
    for episode in range(1, num_test_episodes + 1):
        state = env._reset()
        visualizer.reset()
        visualizer.update_info(episode=episode, steps=0, reward=0, status="Testing")
        
        from util import flatten_and_reshape
        state = flatten_and_reshape(state, num_discs, agent)
        
        total_reward = 0
        success = False
        
        for step in range(500):
            # Select action (no exploration)
            action = agent.act(state)
            from_rod, to_rod = env.decode_action(action)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            next_state_flat = flatten_and_reshape(next_state, num_discs, agent)
            
            # Update visualization
            visualizer.update_state(next_state, from_rod, to_rod, animate=True)
            visualizer.update_info(steps=step + 1, reward=total_reward)
            
            state = next_state_flat
            total_reward += reward
            
            if done:
                success = (len(env.state[2]) == num_discs)
                if success:
                    successes += 1
                    total_steps += (step + 1)
                break
        
        status = "✓ Success!" if success else "✗ Failed"
        visualizer.update_info(status=status)
        print(f"Episode {episode}: {status} | Steps: {step + 1} | Reward: {total_reward:.1f}")
    
    # Summary
    success_rate = (successes / num_test_episodes) * 100
    avg_steps = total_steps / successes if successes > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Testing Summary:")
    print(f"  Success Rate: {success_rate:.1f}% ({successes}/{num_test_episodes})")
    print(f"  Average Steps (successful): {avg_steps:.2f}")
    print(f"  Optimal Steps: {2**num_discs - 1}")
    print(f"{'='*80}")
    
    visualizer.update_info(
        success_rate=success_rate,
        avg_steps=avg_steps,
        status="Testing Complete"
    )
    
    print("\nClose the visualization window to exit.")
    visualizer.root.mainloop()


def train_model(num_discs=3, episodes=1000, animation_speed=0.3, show_every_n=10):
    """
    Train a new model with visualization.
    
    Args:
        num_discs: Number of discs
        episodes: Total training episodes
        animation_speed: Speed of animation
        show_every_n: Show visualization every N episodes
    """
    print("="*80)
    print("Tower of Hanoi - Training Mode")
    print("="*80)
    print(f"Configuration:")
    print(f"  Number of discs: {num_discs}")
    print(f"  Total episodes: {episodes}")
    print(f"  Animation speed: {animation_speed}s")
    print(f"  Visualize every: {show_every_n} episodes")
    print("="*80)
    
    # Initialize environment
    env = TowerOfHanoiEnv(num_discs=num_discs)
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(state_size, action_size)
    
    # Initialize visualizer
    visualizer = TowerOfHanoiVisualizer(num_discs=num_discs, animation_speed=animation_speed)
    
    # Initialize model manager
    model_manager = ModelManager(models_dir='models')
    
    # Start training
    try:
        metrics = train_with_visualization(
            env, agent, visualizer, num_discs,
            total_episodes=episodes,
            eval_interval=50,
            checkpoint_interval=100,
            model_manager=model_manager,
            show_every_n=show_every_n
        )
        
        # Keep window open after training
        visualizer.update_info(status="Training Complete!")
        print("\nTraining finished. Close the visualization window to exit.")
        visualizer.root.mainloop()
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        visualizer.close()
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        visualizer.close()


def demo_visualizer(num_discs=3):
    """
    Demo the visualizer with the optimal solution.
    
    Args:
        num_discs: Number of discs
    """
    print("="*80)
    print("Tower of Hanoi - Visualizer Demo (Optimal Solution)")
    print("="*80)
    
    visualizer = TowerOfHanoiVisualizer(num_discs=num_discs, animation_speed=0.5)
    
    def solve_hanoi(n, source, destination, auxiliary, state, moves):
        """Generate optimal solution moves."""
        if n == 1:
            moves.append((source, destination))
            return
        solve_hanoi(n - 1, source, auxiliary, destination, state, moves)
        moves.append((source, destination))
        solve_hanoi(n - 1, auxiliary, destination, source, state, moves)
    
    # Generate optimal solution
    moves = []
    state = [[i for i in range(num_discs, 0, -1)], [], []]
    solve_hanoi(num_discs, 0, 2, 1, state, moves)
    
    print(f"Optimal solution requires {len(moves)} moves.")
    print("Demonstrating optimal solution...\n")
    
    # Reset state
    state = [[i for i in range(num_discs, 0, -1)], [], []]
    visualizer.update_info(episode=1, steps=0, reward=0, status="Demo - Optimal Solution")
    
    import time
    time.sleep(1)
    
    # Execute moves
    for i, (from_rod, to_rod) in enumerate(moves):
        disc = state[from_rod].pop()
        state[to_rod].append(disc)
        
        visualizer.update_state(state, from_rod, to_rod, animate=True)
        visualizer.update_info(steps=i+1, reward=-(i+1), status=f"Move {i+1}/{len(moves)}")
    
    visualizer.update_info(status="✓ Complete!")
    print(f"\nDemo complete! Solved in {len(moves)} moves (optimal).")
    print("Close the visualization window to exit.")
    visualizer.root.mainloop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Tower of Hanoi - Reinforcement Learning with Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --discs 3 --episodes 1000
  
  # Test a trained model
  python main.py test --model models/model_v1.h5
  
  # Demo the visualizer with optimal solution
  python main.py demo --discs 4
  
  # Quick training with more visualization
  python main.py train --discs 3 --episodes 500 --show-every 5 --speed 0.2
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--discs', type=int, default=3,
                             help='Number of discs (default: 3)')
    train_parser.add_argument('--episodes', type=int, default=1000,
                             help='Number of training episodes (default: 1000)')
    train_parser.add_argument('--speed', type=float, default=0.3,
                             help='Animation speed in seconds (default: 0.3)')
    train_parser.add_argument('--show-every', type=int, default=10,
                             help='Show visualization every N episodes (default: 10)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a trained model')
    test_parser.add_argument('--model', type=str, default=None,
                            help='Path to model file (default: best model in models/)')
    test_parser.add_argument('--discs', type=int, default=3,
                            help='Number of discs (default: 3)')
    test_parser.add_argument('--speed', type=float, default=0.5,
                            help='Animation speed in seconds (default: 0.5)')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Demo the visualizer with optimal solution')
    demo_parser.add_argument('--discs', type=int, default=3,
                            help='Number of discs (default: 3)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(
            num_discs=args.discs,
            episodes=args.episodes,
            animation_speed=args.speed,
            show_every_n=args.show_every
        )
    
    elif args.command == 'test':
        # Find best model if not specified
        model_path = args.model
        if model_path is None:
            models_dir = 'models'
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
                if model_files:
                    # Get the latest model
                    model_files.sort()
                    model_path = os.path.join(models_dir, model_files[-1])
                    print(f"Using latest model: {model_path}")
        
        test_model(
            num_discs=args.discs,
            model_path=model_path,
            animation_speed=args.speed
        )
    
    elif args.command == 'demo':
        demo_visualizer(num_discs=args.discs)
    
    else:
        parser.print_help()
        print("\n" + "="*80)
        print("Quick Start:")
        print("="*80)
        print("1. Demo the visualizer:")
        print("   python main.py demo")
        print("\n2. Train a model:")
        print("   python main.py train --episodes 500")
        print("\n3. Test the trained model:")
        print("   python main.py test")
        print("="*80)


if __name__ == "__main__":
    main()
