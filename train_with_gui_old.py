"""
Enhanced training script for Tower of Hanoi with GUI visualization and comprehensive metrics.
Includes model checkpointing and automatic model replacement based on performance.
"""
import numpy as np
import os
import json
import time
from datetime import datetime
from toh import TowerOfHanoiEnv
from dqn_agent import DQNAgent
from util import flatten_and_reshape
from visualizer import TowerOfHanoiVisualizer


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
    
    def add_episode(self, episode, reward, steps, success, epsilon):
        """Add metrics for an episode."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.steps.append(steps)
        self.successes.append(1 if success else 0)
        self.epsilons.append(epsilon)
    
    def get_recent_stats(self, window=100):
        """Get statistics for recent episodes."""
        if len(self.episodes) == 0:
            return 0.0, 0.0, 0.0
        
        recent_rewards = self.rewards[-window:]
        recent_steps = self.steps[-window:]
        recent_successes = self.successes[-window:]
        
        avg_reward = np.mean(recent_rewards)
        avg_steps = np.mean([s for s, succ in zip(recent_steps, recent_successes) if succ]) if any(recent_successes) else 0
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
        self.performance_history = []
    
    def save_model(self, agent, metrics, version=None):
        """Save a model with its performance metrics."""
        if version is None:
            version = self.current_version
            self.current_version += 1
        
        model_path = os.path.join(self.models_dir, f'model_v{version}.h5')
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
    
    def load_best_model(self, agent):
        """Load the best performing model."""
        if self.best_version == 0 and self.current_version == 0:
            return None
        
        model_path = os.path.join(self.models_dir, f'model_v{self.best_version}.h5')
        if os.path.exists(model_path):
            agent.load(model_path)
            return model_path
        return None
    
    def should_replace_model(self, current_metrics, patience=3):
        """
        Determine if current model should be replaced with best model.
        
        Args:
            current_metrics: Current model performance metrics
            patience: Number of evaluations to wait before replacement
        """
        self.performance_history.append(current_metrics)
        
        # Keep only recent history
        if len(self.performance_history) > patience:
            self.performance_history.pop(0)
        
        # Not enough history yet
        if len(self.performance_history) < patience:
            return False
        
        # Check if we have a best model
        if self.best_performance['success_rate'] == 0.0:
            return False
        
        # Check if recent performance is consistently worse
        recent_success_rate = np.mean([m['success_rate'] for m in self.performance_history])
        recent_avg_steps = np.mean([m['avg_steps'] for m in self.performance_history if m['avg_steps'] > 0])
        
        # Replace if success rate dropped significantly or steps increased significantly
        success_drop = self.best_performance['success_rate'] - recent_success_rate > 20
        steps_increase = (recent_avg_steps > self.best_performance['avg_steps'] * 1.5 
                         if recent_avg_steps > 0 and self.best_performance['avg_steps'] < float('inf') 
                         else False)
        
        return success_drop or steps_increase
    
    def update_best_model(self, agent, metrics):
        """Update the best model if current performance is better."""
        improved = False
        
        # Primary metric: success rate
        if metrics['success_rate'] > self.best_performance['success_rate']:
            improved = True
        # Secondary metric: average steps (if success rate is similar)
        elif (abs(metrics['success_rate'] - self.best_performance['success_rate']) < 5 and
              metrics['avg_steps'] > 0 and
              metrics['avg_steps'] < self.best_performance['avg_steps']):
            improved = True
        
        if improved:
            self.best_version = self.current_version
            self.best_performance = metrics.copy()
            self.performance_history = []  # Reset history
            model_path = self.save_model(agent, metrics)
            return True, model_path
        
        return False, None


def train_with_visualization(env, agent, visualizer, num_discs, total_episodes=1000,
                            eval_interval=50, checkpoint_interval=100,
                            model_manager=None, show_every_n=10):
    """
    Train the agent with GUI visualization and comprehensive metrics tracking.
    
    Args:
        env: Tower of Hanoi environment
        agent: DQN agent
        visualizer: GUI visualizer
        num_discs: Number of discs
        total_episodes: Total number of training episodes
        eval_interval: Episodes between performance evaluations
        checkpoint_interval: Episodes between model checkpoints
        model_manager: Model manager for versioning
        show_every_n: Show visualization every N episodes
    """
    metrics = TrainingMetrics()
    
    for episode in range(1, total_episodes + 1):
        if visualizer.is_closed():
            print("Visualizer window closed. Stopping training.")
            break
        
        # Reset environment
        state = env._reset()
        state = flatten_and_reshape(state, num_discs, agent)
        total_reward = 0
        success = False
        
        # Determine if we should visualize this episode
        visualize_episode = (episode % show_every_n == 0) or (episode <= 5)
        
        if visualize_episode:
            visualizer.reset()
            visualizer.update_info(episode=episode, steps=0, reward=0, 
                                 epsilon=agent.epsilon, status="Training")
        
        # Run episode
        for step in range(500):  # Max steps per episode
            # Select action
            action = agent.act(state)
            
            # Get move details for visualization
            from_rod, to_rod = env.decode_action(action)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            next_state_flat = flatten_and_reshape(next_state, num_discs, agent)
            
            # Remember experience
            agent.remember(state, action, reward, next_state_flat, done)
            
            # Update state and reward
            state = next_state_flat
            total_reward += reward
            
            # Visualize if needed
            if visualize_episode:
                visualizer.update_state(next_state, from_rod, to_rod, animate=True)
                visualizer.update_info(steps=step + 1, reward=total_reward)
            
            if done:
                success = (len(env.state[2]) == num_discs)
                break
        
        # Train agent
        agent.replay()
        
        # Record metrics
        metrics.add_episode(episode, total_reward, step + 1, success, agent.epsilon)
        
        # Get recent statistics
        avg_reward, avg_steps, success_rate = metrics.get_recent_stats(window=100)
        
        # Update visualization
        status = "✓ Success!" if success else "Failed"
        visualizer.update_info(
            epsilon=agent.epsilon,
            success_rate=success_rate,
            avg_steps=avg_steps if avg_steps > 0 else 0,
            status=status
        )
        
        # Print progress
        print(f"Episode {episode}/{total_episodes} | Steps: {step + 1} | "
              f"Reward: {total_reward:.1f} | Success: {success} | "
              f"Epsilon: {agent.epsilon:.3f} | Success Rate: {success_rate:.1f}%")
        
        # Evaluate and checkpoint
        if episode % eval_interval == 0:
            print(f"\n{'='*80}")
            print(f"Evaluation at Episode {episode}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Steps (last 100): {avg_steps:.2f}")
            print(f"  Success Rate (last 100): {success_rate:.1f}%")
            print(f"{'='*80}\n")
            
            # Check if model should be replaced
            if model_manager:
                current_metrics = {
                    'success_rate': success_rate,
                    'avg_steps': avg_steps,
                    'avg_reward': avg_reward,
                    'epsilon': agent.epsilon
                }
                
                # Update best model if improved
                improved, model_path = model_manager.update_best_model(agent, current_metrics)
                if improved:
                    print(f"✓ New best model saved! Success Rate: {success_rate:.1f}%, "
                          f"Avg Steps: {avg_steps:.2f}")
                    visualizer.update_info(status="New Best Model!")
                
                # Check if we should replace current model with best
                if model_manager.should_replace_model(current_metrics):
                    print(f"⚠ Performance degraded. Loading best model (v{model_manager.best_version})...")
                    model_manager.load_best_model(agent)
                    # Reset epsilon to encourage exploration
                    agent.epsilon = min(0.5, agent.epsilon * 2)
                    visualizer.update_info(status="Model Replaced!")
        
        # Save checkpoint
        if model_manager and episode % checkpoint_interval == 0:
            checkpoint_metrics = {
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'avg_reward': avg_reward,
                'epsilon': agent.epsilon
            }
            model_manager.save_model(agent, checkpoint_metrics)
            print(f"Checkpoint saved at episode {episode}")
    
    # Save final metrics
    metrics.save_to_file('training_metrics.json')
    print("\nTraining completed! Metrics saved to training_metrics.json")
    
    return metrics


def main():
    """Main training function."""
    # Configuration
    num_discs = 3
    total_episodes = 1000
    animation_speed = 0.3  # Seconds per move
    show_every_n = 10  # Visualize every 10th episode
    
    print("="*80)
    print("Tower of Hanoi - Reinforcement Learning Training")
    print("="*80)
    print(f"Configuration:")
    print(f"  Number of discs: {num_discs}")
    print(f"  Total episodes: {total_episodes}")
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
            total_episodes=total_episodes,
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


if __name__ == "__main__":
    main()
