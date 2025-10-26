"""
Model Testing and Comparison Tool
Load and test trained models, compare performance
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from toh import TowerOfHanoiEnv
from dqn_agent import DQNAgent
from model_evaluation import ModelEvaluator, LearningRateTracker
from visualizer import create_visualizer
from util import flatten_state
from PyQt6.QtWidgets import QApplication
import time


def load_and_test_model(model_path, num_discs=3, visualize=False, num_episodes=100):
    """Load a model and test its performance"""
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_path}")
    print(f"{'='*80}\n")
    
    # Initialize environment
    env = TowerOfHanoiEnv(num_discs=num_discs)
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n
    
    # Initialize agent and load model
    agent = DQNAgent(state_size, action_size)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None
    
    agent.load(model_path)
    print(f"✓ Model loaded successfully\n")
    
    # Create evaluator
    evaluator = ModelEvaluator(env, num_discs=num_discs, optimal_steps=7)
    
    # Run evaluation
    print(f"Running evaluation on {num_episodes} episodes...")
    eval_report = evaluator.evaluate_model(agent, num_episodes=num_episodes, verbose=True)
    
    # Generate and print report
    model_name = Path(model_path).stem
    evaluator.generate_report(eval_report, model_name=model_name)
    
    # Visualize if requested
    if visualize:
        print("\nStarting visualization of model performance...")
        visualize_model(agent, env, num_discs)
    
    return eval_report


def visualize_model(agent, env, num_discs=3, num_demos=5):
    """Visualize model playing Tower of Hanoi"""
    visualizer, app = create_visualizer(env, num_discs=num_discs)
    
    print(f"\nDemonstrating {num_demos} episodes...")
    
    for demo in range(num_demos):
        print(f"\nDemo {demo + 1}/{num_demos}")
        state = env._reset()
        visualizer.update_state(env.state)
        visualizer.update_info(episode=demo + 1, step=0, reward=0)
        time.sleep(1.0)  # Pause to show initial state
        
        done = False
        steps = 0
        total_reward = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            # Get action from agent
            flat_state = flatten_state(state, num_discs)
            flat_state = np.reshape(flat_state, [1, agent.state_size])
            action = np.argmax(agent.model.predict(flat_state, verbose=0)[0])
            
            # Get move details
            from_rod, to_rod = env.decode_action(action)
            disc = env.state[from_rod][-1] if env.state[from_rod] else None
            
            # Execute
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            # Animate
            if disc:
                visualizer.animate_move(from_rod, to_rod, disc)
                visualizer.update_state(env.state)
                visualizer.update_info(step=steps, reward=total_reward)
        
        success = (len(env.state[2]) == num_discs)
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status} in {steps} steps (Reward: {total_reward:.1f})")
        time.sleep(2.0)
    
    print("\nVisualization complete. Close the window to exit.")
    sys.exit(app.exec())


def compare_models(model_paths, num_discs=3, num_episodes=100):
    """Compare multiple models"""
    print(f"\n{'='*80}")
    print(f"COMPARING {len(model_paths)} MODELS")
    print(f"{'='*80}\n")
    
    reports = []
    model_names = []
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Warning: Model not found: {model_path}")
            continue
        
        report = load_and_test_model(model_path, num_discs, visualize=False, num_episodes=num_episodes)
        if report:
            reports.append(report)
            model_names.append(Path(model_path).stem)
    
    if len(reports) > 1:
        env = TowerOfHanoiEnv(num_discs=num_discs)
        evaluator = ModelEvaluator(env, num_discs=num_discs, optimal_steps=7)
        evaluator.compare_models(reports, model_names)
    else:
        print("\nNeed at least 2 valid models to compare")


def list_available_models():
    """List all available models"""
    models_dir = Path("models")
    if not models_dir.exists():
        print("No models directory found")
        return []
    
    model_files = sorted(models_dir.glob("*.weights.h5"))
    
    print(f"\n{'='*80}")
    print("AVAILABLE MODELS")
    print(f"{'='*80}\n")
    
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file.name}")
        
        # Try to load metadata
        metadata_file = model_file.with_name(model_file.name.replace('.weights.h5', '_metadata.json'))
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"   Success Rate: {metadata.get('success_rate', 'N/A'):.2f}%")
            print(f"   Avg Steps: {metadata.get('avg_steps', 'N/A'):.2f}")
            print(f"   Efficiency: {metadata.get('efficiency_score', 'N/A'):.2f}%")
        print()
    
    return list(model_files)


def main():
    parser = argparse.ArgumentParser(description="Test and compare trained models")
    parser.add_argument('--model', type=str, help='Path to model file to test')
    parser.add_argument('--compare', nargs='+', help='Paths to models to compare')
    parser.add_argument('--list', action='store_true', help='List all available models')
    parser.add_argument('--visualize', action='store_true', help='Visualize model performance')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes for evaluation')
    parser.add_argument('--num-discs', type=int, default=3, help='Number of discs')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    elif args.compare:
        compare_models(args.compare, args.num_discs, args.episodes)
    elif args.model:
        load_and_test_model(args.model, args.num_discs, args.visualize, args.episodes)
    else:
        # Interactive mode
        models = list_available_models()
        if not models:
            print("No models found. Train a model first!")
            return
        
        print("\nSelect a model to test (or 'c' to compare, 'q' to quit):")
        choice = input("> ").strip()
        
        if choice.lower() == 'q':
            return
        elif choice.lower() == 'c':
            print("\nEnter model numbers to compare (space-separated):")
            indices = input("> ").strip().split()
            try:
                selected = [models[int(i)-1] for i in indices]
                compare_models([str(m) for m in selected], args.num_discs, args.episodes)
            except (ValueError, IndexError):
                print("Invalid selection")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    print("\nVisualize? (y/n)")
                    viz = input("> ").strip().lower() == 'y'
                    load_and_test_model(str(models[idx]), args.num_discs, viz, args.episodes)
                else:
                    print("Invalid model number")
            except ValueError:
                print("Invalid input")


if __name__ == "__main__":
    main()
