"""
Enhanced training script for Tower of Hanoi with PyQt6 GUI visualization.
Includes model checkpointing, evaluation, and learning rate tracking.
"""
import numpy as np
import os
import json
import time
import sys
from datetime import datetime
from PyQt6.QtWidgets import QApplication
from toh import TowerOfHanoiEnv
from dqn_agent import DQNAgent
from util import flatten_state
from visualizer import create_visualizer
from model_evaluation import ModelEvaluator, LearningRateTracker


class TrainingMetrics:
    """Track and manage training metrics."""
    
    def __init__(self):
        self.episodes = []
        self.rewards = []
        self.steps = []
        self.successes = []
        self.epsilons = []
        self.best_avg_steps = float('inf')
        self.best_success_rate = 0.0
        self.history = []  # Complete episode history for learning progress tracking
    
    def add_episode(self, episode, reward, steps, success, epsilon):
        """Add metrics for an episode."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.steps.append(steps)
        self.successes.append(1 if success else 0)
        self.epsilons.append(epsilon)
        
        # Add to history for learning progress tracking
        self.history.append({
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'success': success,
            'epsilon': epsilon
        })
    
    def get_recent_stats(self, window=100):
        """Get statistics for recent episodes."""
        if len(self.episodes) == 0:
            return 0.0, 0.0, 0.0
        
        recent_rewards = self.rewards[-window:]
        recent_steps = self.steps[-window:]
        recent_successes = self.successes[-window:]
        
        avg_reward = np.mean(recent_rewards)
        successful_steps = [s for s, succ in zip(recent_steps, recent_successes) if succ]
        avg_steps = np.mean(successful_steps) if successful_steps else 0
        success_rate = np.mean(recent_successes) * 100
        
        return avg_reward, avg_steps, success_rate
    
    def save_to_file(self, filename):
        """Save metrics to a JSON file."""
        data = {
            'episodes': self.episodes,
            'rewards': self.rewards,
            'steps': self.steps,
            'successes': self.successes,
            'epsilons': self.epsilons,
            'best_avg_steps': self.best_avg_steps,
            'best_success_rate': self.best_success_rate
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


class ModelManager:
    """Manage model versions and automatic replacement."""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.current_version = 0
        self.best_version = 0
        self.best_performance = {'success_rate': 0.0, 'avg_steps': float('inf')}
    
    def save_model(self, agent, metrics, version=None):
        """Save a model with its performance metrics."""
        if version is None:
            version = self.current_version
            self.current_version += 1
        
        model_path = os.path.join(self.models_dir, f'model_v{version}.weights.h5')
        agent.save(model_path)
        
        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'success_rate': metrics['success_rate'],
            'avg_steps': metrics['avg_steps'],
            'avg_reward': metrics['avg_reward'],
            'epsilon': metrics['epsilon']
        }
        
        metadata_path = os.path.join(self.models_dir, f'model_v{version}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def should_replace_model(self, agent, metrics):
        """Check if current model should replace the best model."""
        # Need significant data to make decision
        if len(metrics.episodes) < 100:
            return False, None
        
        current_metrics = {
            'success_rate': metrics.get_recent_stats()[2],
            'avg_steps': metrics.get_recent_stats()[1],
            'avg_reward': metrics.get_recent_stats()[0],
            'epsilon': agent.epsilon
        }
        
        improved = False
        
        # Primary: success rate improvement
        if current_metrics['success_rate'] > self.best_performance['success_rate'] + 5:
            improved = True
        # Secondary: steps improvement with similar success rate
        elif (abs(current_metrics['success_rate'] - self.best_performance['success_rate']) < 5 and
              current_metrics['avg_steps'] > 0 and
              current_metrics['avg_steps'] < self.best_performance['avg_steps'] * 0.9):
            improved = True
        
        if improved:
            self.best_version = self.current_version
            self.best_performance = current_metrics.copy()
            model_path = self.save_model(agent, current_metrics)
            
            # Also save as best model
            best_path = os.path.join(self.models_dir, 'best_model.weights.h5')
            agent.save(best_path)
            
            return True, model_path
        
        return False, None


def train_with_visualization(num_discs=3, total_episodes=1000, batch_size=32, 
                            show_every_n=10, eval_interval=50, checkpoint_interval=100,
                            return_on_complete=False):
    """
    Train a DQN agent with PyQt6 GUI visualization.
    
    Args:
        num_discs: Number of discs in the puzzle
        total_episodes: Total training episodes
        batch_size: Batch size for training
        show_every_n: Update visualization every N episodes
        eval_interval: Run evaluation every N episodes
        checkpoint_interval: Save checkpoint every N episodes
        return_on_complete: If True, return after training instead of calling sys.exit()
                           (useful when called from another Qt application)
    """
    print("="*80)
    print("Initializing training...")
    print("="*80)
    
    # Initialize environment
    env = TowerOfHanoiEnv(num_discs=num_discs)
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(state_size, action_size)
    
    # Create visualizer
    visualizer, app = create_visualizer(env, num_discs=num_discs)
    
    # Initialize managers
    metrics = TrainingMetrics()
    model_manager = ModelManager()
    evaluator = ModelEvaluator(env, num_discs=num_discs, optimal_steps=7)
    learning_tracker = LearningRateTracker()
    
    # Track training session
    training_start_time = time.time()
    session_data = {
        'session_name': f"Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now().isoformat(),
        'num_discs': num_discs,
        'total_episodes': total_episodes,
        'batch_size': batch_size,
        'state_size': state_size,
        'action_size': action_size
    }
    
    print(f"Training for {total_episodes} episodes...")
    print(f"State size: {state_size}, Action size: {action_size}")
    print("="*80 + "\n")
    
    # Training loop in background thread
    def training_loop():
        for episode in range(1, total_episodes + 1):
            if visualizer.should_stop:
                print("\nTraining stopped by user.")
                break
            
            visualizer.wait_if_paused()
            
            # Reset environment
            state = env._reset()
            total_reward = 0
            success = False
            step = 0
            max_steps = 500
            
            # Determine if we should visualize this episode
            # Always visualize when show_visualization is True, otherwise skip
            visualize_episode = visualizer.show_visualization and (episode % show_every_n == 0)
            
            if visualize_episode:
                visualizer.update_state(env.state)
                visualizer.update_info(episode=episode, step=0, reward=0, epsilon=agent.epsilon)
                import time
                time.sleep(0.5)  # Pause to show initial state
            
            # Run episode
            while step < max_steps:
                visualizer.wait_if_paused()
                
                if visualizer.should_stop:
                    break
                
                # Select action
                flat_state = flatten_state(state, num_discs)
                flat_state = np.reshape(flat_state, [1, agent.state_size])
                action = agent.act(flat_state)
                
                # Get move details for visualization
                from_rod, to_rod = env.decode_action(action)
                disc = env.state[from_rod][-1] if env.state[from_rod] else None
                
                # Execute action
                next_state, reward, done, _ = env.step(action)
                
                # Remember experience
                next_flat_state = flatten_state(next_state, num_discs)
                next_flat_state = np.reshape(next_flat_state, [1, agent.state_size])
                agent.remember(flat_state, action, reward, next_flat_state, done)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                step += 1
                
                # Visualize if needed
                if visualize_episode and disc:
                    visualizer.animate_move(from_rod, to_rod, disc)
                    visualizer.update_state(env.state)
                    visualizer.update_info(step=step, reward=total_reward)
                
                if done:
                    success = (len(env.state[2]) == num_discs)
                    break
            
            # Train agent with experience replay
            if len(agent.memory) > batch_size:
                agent.replay()
            
            # Record metrics
            metrics.add_episode(episode, total_reward, step, success, agent.epsilon)
            
            # Get recent statistics
            avg_reward, avg_steps, success_rate = metrics.get_recent_stats(window=100)
            
            # Update visualization with stats - ALWAYS update, not just when visualizing
            visualizer.update_info(
                episode=episode,
                epsilon=agent.epsilon,
                success_rate=success_rate
            )
            
            # Print progress
            if episode % 10 == 0:
                status = "✓" if success else "✗"
                print(f"Episode {episode:4d} {status} | Steps: {step:3d} | Reward: {total_reward:6.1f} | "
                      f"Epsilon: {agent.epsilon:.3f} | Success Rate: {success_rate:5.1f}% | "
                      f"Avg Steps: {avg_steps:.1f}")
            
            # Checkpoint
            if episode % checkpoint_interval == 0:
                improved, model_path = model_manager.should_replace_model(agent, metrics)
                if improved:
                    print(f"\n🎉 New best model saved! Success rate: {success_rate:.1f}%, Avg steps: {avg_steps:.1f}")
                    print(f"   Saved to: {model_path}\n")
            
            # Evaluation
            if episode % eval_interval == 0:
                print(f"\n{'='*80}")
                print(f"Evaluation at Episode {episode}:")
                print(f"  Recent Success Rate: {success_rate:.1f}%")
                print(f"  Recent Avg Steps: {avg_steps:.1f}")
                print(f"  Recent Avg Reward: {avg_reward:.1f}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                
                # Quick evaluation
                eval_report = evaluator.evaluate_model(agent, num_episodes=50, verbose=False)
                print(f"  Eval Success Rate: {eval_report['success_rate']:.1f}%")
                print(f"  Eval Efficiency: {eval_report['efficiency_score']:.1f}%")
                print(f"  Optimal Solves: {eval_report['optimal_solves']}/50")
                print(f"{'='*80}\n")
        
        # Training completed
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        print(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
        print(f"Episodes Completed: {total_episodes}")
        print(f"Time per Episode: {(total_training_time/total_episodes)*1000:.2f}ms")
        print("="*80 + "\n")
        
        # Final comprehensive evaluation
        print("Running final comprehensive evaluation...")
        final_eval_report = evaluator.evaluate_model(agent, num_episodes=100, verbose=True)
        evaluator.generate_report(final_eval_report, model_name="Final Model")
        
        # Save final model with evaluation metadata
        final_metrics = {
            'success_rate': final_eval_report['success_rate'],
            'avg_steps': final_eval_report['avg_steps'],
            'efficiency_score': final_eval_report['efficiency_score'],
            'optimal_solves': final_eval_report['optimal_solves'],
            'avg_reward': final_eval_report['avg_reward'],
            'epsilon': agent.epsilon,
            'training_episodes': total_episodes,
            'training_time': total_training_time
        }
        final_path = model_manager.save_model(agent, final_metrics)
        
        # Save evaluation report
        eval_report_path = final_path.replace('.weights.h5', '_evaluation.json')
        evaluator.save_report(final_eval_report, eval_report_path)
        
        # Track learning progress
        learning_metrics = evaluator.track_learning_progress(metrics, total_episodes)
        if learning_metrics:
            print("\nLEARNING PROGRESS ANALYSIS")
            print("="*80)
            print(f"Episodes Trained: {learning_metrics['episodes_trained']}")
            print(f"Convergence: {learning_metrics['convergence_time']}")
            for window, data in learning_metrics['learning_windows'].items():
                print(f"\n{window} episodes:")
                print(f"  Success Rate: {data['success_rate']:.2f}%")
                print(f"  Avg Steps: {data['avg_steps']:.2f}")
            print("="*80 + "\n")
        
        # Save complete training session
        session_data.update({
            'total_training_time': total_training_time,
            'final_evaluation': final_eval_report,
            'learning_metrics': learning_metrics,
            'final_model_path': final_path,
            'convergence_episode': learning_metrics['convergence_episode'] if learning_metrics else None
        })
        
        session_file = learning_tracker.save_training_session(session_data)
        print(f"Training session report saved to: {session_file}\n")
        
        # Save metrics
        metrics_file = os.path.join('models', f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        metrics.save_to_file(metrics_file)
        
        print(f"\n{'='*80}")
        print("Training Complete!")
        print(f"{'='*80}")
        print(f"Final Success Rate: {final_metrics['success_rate']:.1f}%")
        print(f"Final Avg Steps: {final_metrics['avg_steps']:.1f}")
        print(f"Final model saved to: {final_path}")
        print(f"Metrics saved to: {metrics_file}")
        print(f"{'='*80}\n")
    
    # Run training in background thread
    from threading import Thread
    training_thread = Thread(target=training_loop, daemon=True)
    training_thread.start()
    
    # Start Qt event loop (only if not already running)
    if return_on_complete:
        # Wait for training to complete without blocking the existing event loop
        import time as time_module
        while training_thread.is_alive():
            app.processEvents()
            time_module.sleep(0.1)
    else:
        # Standalone mode - start event loop
        sys.exit(app.exec())


if __name__ == "__main__":
    train_with_visualization(num_discs=3, total_episodes=1000, show_every_n=20)
