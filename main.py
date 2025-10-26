"""
Main launcher for Tower of Hanoi RL Training with PyQt6 GUI Visualization.
"""
import argparse
import sys
import os
import time
import numpy as np
from PyQt6.QtWidgets import QApplication
from toh import TowerOfHanoiEnv
from dqn_agent import DQNAgent
from visualizer import create_visualizer


def demo_visualizer(num_discs=3, return_on_complete=False):
    """
    Demo the visualizer with optimal solution.
    
    Args:
        num_discs: Number of discs
        return_on_complete: If True, return after demo instead of calling sys.exit()
    """
    print("="*80)
    print("Tower of Hanoi - Visualizer Demo (Optimal Solution)")
    print("="*80)
    
    # Initialize environment
    env = TowerOfHanoiEnv(num_discs=num_discs)
    
    # Create visualizer
    visualizer, app = create_visualizer(env, num_discs=num_discs)
    
    def solve_hanoi(n, source, destination, auxiliary, moves):
        """Generate optimal solution moves."""
        if n == 1:
            moves.append((source, destination))
            return
        solve_hanoi(n - 1, source, auxiliary, destination, moves)
        moves.append((source, destination))
        solve_hanoi(n - 1, auxiliary, destination, source, moves)
    
    # Generate optimal solution
    moves = []
    solve_hanoi(num_discs, 0, 2, 1, moves)
    
    print(f"Optimal solution requires {len(moves)} moves.")
    print("Demonstrating optimal solution...\n")
    
    # Reset state
    state = [[i for i in range(num_discs, 0, -1)], [], []]
    visualizer.update_state(state)
    visualizer.update_info(episode=1, step=0, reward=0)
    
    # Execute moves in background thread
    def run_demo():
        time.sleep(1)
        
        for i, (from_rod, to_rod) in enumerate(moves):
            if visualizer.should_stop:
                break
            
            # Get disc
            disc = state[from_rod][-1]
            
            # Remove from source
            state[from_rod].pop()
            
            # Animate move
            visualizer.animate_move(from_rod, to_rod, disc)
            
            # Add to destination
            state[to_rod].append(disc)
            visualizer.update_state(state)
            visualizer.update_info(step=i+1, reward=-(i+1))
            
            time.sleep(0.3)
        
        print(f"\nDemo complete! Solved in {len(moves)} moves (optimal).")
    
    from threading import Thread
    demo_thread = Thread(target=run_demo, daemon=True)
    demo_thread.start()
    
    # Start Qt event loop (only if not already running)
    if return_on_complete:
        # Wait for demo to complete without blocking the existing event loop
        import time as time_module
        while demo_thread.is_alive():
            app.processEvents()
            time_module.sleep(0.1)
    else:
        # Standalone mode - start event loop
        sys.exit(app.exec())


def test_model(num_discs=3, model_path=None, return_on_complete=False):
    """
    Test a trained model with visualization.
    
    Args:
        num_discs: Number of discs
        model_path: Path to trained model file
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
    
    # Create visualizer
    visualizer, app = create_visualizer(env, num_discs=num_discs)
    
    # Test for multiple episodes
    num_test_episodes = 5
    successes = 0
    total_steps_all = 0
    
    def run_test():
        nonlocal successes, total_steps_all
        
        for episode in range(1, num_test_episodes + 1):
            if visualizer.should_stop:
                break
            
            visualizer.wait_if_paused()
            
            state = env._reset()
            visualizer.update_state(env.state)
            visualizer.update_info(episode=episode, step=0, reward=0)
            
            total_reward = 0
            done = False
            step = 0
            max_steps = 1000
            
            while not done and step < max_steps and not visualizer.should_stop:
                visualizer.wait_if_paused()
                
                from util import flatten_state
                flat_state = flatten_state(state, num_discs)
                flat_state = np.reshape(flat_state, [1, agent.state_size])
                action = agent.act(flat_state)
                
                # Get move details before step
                from_rod, to_rod = env.decode_action(action)
                disc = env.state[from_rod][-1] if env.state[from_rod] else None
                
                next_state, reward, done, _ = env.step(action)
                
                total_reward += reward
                step += 1
                
                # Animate move if valid
                if disc:
                    visualizer.animate_move(from_rod, to_rod, disc)
                
                visualizer.update_state(env.state)
                visualizer.update_info(step=step, reward=total_reward)
                
                state = next_state
                
                if done:
                    success = (len(env.state[2]) == num_discs)
                    if success:
                        successes += 1
                    total_steps_all += step
                    
                    status = "✓ Success!" if success else "✗ Failed"
                    print(f"Episode {episode}: {status} | Steps: {step} | Reward: {total_reward:.1f}")
                    time.sleep(1)
        
        # Summary
        success_rate = (successes / num_test_episodes) * 100
        avg_steps = total_steps_all / num_test_episodes
        
        print(f"\n{'='*80}")
        print(f"Testing Complete:")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Average Steps: {avg_steps:.1f}")
        print(f"{'='*80}")
    
    from threading import Thread
    test_thread = Thread(target=run_test, daemon=True)
    test_thread.start()
    
    sys.exit(app.exec())


def train_model(num_discs=3, episodes=1000, batch_size=32):
    """
    Train a new model with visualization.
    
    Args:
        num_discs: Number of discs
        episodes: Number of training episodes
        batch_size: Batch size for experience replay
    """
    print("="*80)
    print("Tower of Hanoi - Model Training")
    print("="*80)
    
    from train_with_gui import train_with_visualization
    
    train_with_visualization(
        num_discs=num_discs,
        total_episodes=episodes,
        batch_size=batch_size
    )


def main():
    """Main entry point with interactive menu."""
    parser = argparse.ArgumentParser(description="Tower of Hanoi RL Trainer")
    parser.add_argument('command', nargs='?', choices=['train', 'test', 'demo', 'cli'],
                       help='Command to run (default: launch GUI)')
    parser.add_argument('--discs', type=int, default=3,
                       help='Number of discs (default: 3)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file for testing')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--cli', action='store_true',
                       help='Force CLI menu instead of GUI')
    
    args = parser.parse_args()
    
    # Launch GUI by default if no command specified and not --cli
    if not args.command and not args.cli:
        print("\n" + "="*80)
        print("  🗼  Tower of Hanoi - Reinforcement Learning Trainer  🗼")
        print("="*80)
        print("\nLaunching GUI...\n")
        from gui_launcher import main as gui_main
        gui_main()
        return
    
    # Interactive CLI menu if --cli or command='cli'
    if not args.command or args.command == 'cli':
        print("\n" + "="*80)
        print("  🗼  Tower of Hanoi - Reinforcement Learning Trainer  🗼")
        print("="*80)
        print("\nWelcome! This tool will help you train an AI agent to solve Tower of Hanoi.\n")
        
        while True:
            print("What would you like to do?\n")
            print("1. 🎬 Demo - Watch the optimal solution")
            print("2. 🏋️  Train - Train a new AI model")
            print("3. 🧪 Test - Test a trained model")
            print("4. ⚡ Quick Train - Fast training (500 episodes)")
            print("5. 📊 Compare Models - Compare performance of trained models")
            print("6. 📈 Learning Reports - View training session reports")
            print("7. 🎓 Tutorial - Learn how it works")
            print("8. ❌ Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                print("\n" + "="*80)
                print("Running demo with optimal solution...")
                print("="*80 + "\n")
                demo_visualizer(num_discs=args.discs)
                break
            elif choice == '2':
                print("\n" + "="*80)
                print("Starting training...")
                print("="*80 + "\n")
                train_model(num_discs=args.discs, episodes=args.episodes)
                break
            elif choice == '3':
                model_path = args.model or "models/best_model.weights.h5"
                print("\n" + "="*80)
                print(f"Testing model: {model_path}")
                print("="*80 + "\n")
                test_model(num_discs=args.discs, model_path=model_path)
                break
            elif choice == '4':
                print("\n" + "="*80)
                print("Starting quick training (500 episodes)...")
                print("="*80 + "\n")
                train_model(num_discs=args.discs, episodes=500)
                break
            elif choice == '5':
                print("\n" + "="*80)
                print("📊 Comparing Models")
                print("="*80 + "\n")
                from test_model import list_available_models, compare_models
                models = list_available_models()
                if len(models) < 2:
                    print("Need at least 2 models to compare. Train more models first!")
                else:
                    print("Enter model numbers to compare (space-separated, e.g., '1 2 3'):")
                    indices = input("> ").strip().split()
                    try:
                        selected = [str(models[int(i)-1]) for i in indices]
                        compare_models(selected, num_discs=3, num_episodes=100)
                    except (ValueError, IndexError):
                        print("Invalid selection")
                input("\nPress Enter to continue...")
            elif choice == '6':
                print("\n" + "="*80)
                print("📈 Learning Reports")
                print("="*80 + "\n")
                from model_evaluation import LearningRateTracker
                tracker = LearningRateTracker()
                tracker.compare_learning_rates()
                input("\nPress Enter to continue...")
            elif choice == '7':
                print("\n" + "="*80)
                print("�📚 Tower of Hanoi Tutorial")
                print("="*80)
                print("""
The Tower of Hanoi is a classic puzzle with these rules:
1. You have 3 rods (A, B, C) and N discs of different sizes
2. All discs start on rod A, largest at bottom
3. Goal: Move all discs to rod C
4. Rules:
   - Only one disc can be moved at a time
   - A disc can only be placed on top of a larger disc
   - A disc must be moved from the top of one rod to another

The minimal solution requires 2^N - 1 moves.
For 3 discs: 7 moves minimum.

This AI uses Deep Q-Learning (DQN) to learn the optimal strategy!
                """)
                print("="*80 + "\n")
            elif choice == '8':
                print("\n👋 Thanks for using Tower of Hanoi RL Trainer!")
                print("="*80 + "\n")
                sys.exit(0)
            else:
                print("❌ Invalid choice. Please enter 1-8.\n")
    else:
        # Command line mode
        if args.command == 'demo':
            demo_visualizer(num_discs=args.discs)
        elif args.command == 'train':
            train_model(num_discs=args.discs, episodes=args.episodes, batch_size=args.batch_size)
        elif args.command == 'test':
            model_path = args.model or "models/best_model.weights.h5"
            test_model(num_discs=args.discs, model_path=model_path)


if __name__ == "__main__":
    main()
