"""
Model Evaluation and Reporting System
Tracks learning efficiency, accuracy, and provides comparison metrics
"""

import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path


class ModelEvaluator:
    """Evaluates model performance and tracks learning efficiency"""
    
    def __init__(self, env, num_discs=3, optimal_steps=7):
        self.env = env
        self.num_discs = num_discs
        self.optimal_steps = optimal_steps
        self.evaluation_history = []
    
    def evaluate_model(self, agent, num_episodes=100, verbose=False):
        """
        Comprehensive model evaluation
        Returns metrics about model performance
        """
        from util import flatten_state
        
        start_time = time.time()
        
        results = {
            'successes': 0,
            'failures': 0,
            'total_steps': 0,
            'total_rewards': 0,
            'optimal_solves': 0,  # Solved in exactly optimal steps
            'near_optimal': 0,     # Within 20% of optimal
            'step_distribution': [],
            'episode_times': []
        }
        
        for episode in range(num_episodes):
            episode_start = time.time()
            state = self.env._reset()
            done = False
            steps = 0
            episode_reward = 0
            max_steps = 500
            
            while not done and steps < max_steps:
                flat_state = flatten_state(state, self.num_discs)
                flat_state = np.reshape(flat_state, [1, agent.state_size])
                
                # Use greedy action (no exploration during evaluation)
                action = np.argmax(agent.model.predict(flat_state, verbose=0)[0])
                
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                steps += 1
            
            episode_time = time.time() - episode_start
            results['episode_times'].append(episode_time)
            
            # Check if successful
            success = (len(self.env.state[2]) == self.num_discs)
            
            if success:
                results['successes'] += 1
                results['total_steps'] += steps
                results['step_distribution'].append(steps)
                
                if steps == self.optimal_steps:
                    results['optimal_solves'] += 1
                elif steps <= self.optimal_steps * 1.2:
                    results['near_optimal'] += 1
            else:
                results['failures'] += 1
                
            results['total_rewards'] += episode_reward
            
            if verbose and (episode + 1) % 10 == 0:
                success_rate = (results['successes'] / (episode + 1)) * 100
                print(f"  Evaluated {episode + 1}/{num_episodes} episodes - Success: {success_rate:.1f}%")
        
        evaluation_time = time.time() - start_time
        
        # Calculate statistics
        success_rate = (results['successes'] / num_episodes) * 100
        avg_steps = results['total_steps'] / results['successes'] if results['successes'] > 0 else 0
        avg_reward = results['total_rewards'] / num_episodes
        avg_episode_time = np.mean(results['episode_times'])
        
        # Calculate efficiency score (lower is better)
        # Optimal is 100%, efficiency = optimal_steps / avg_steps * 100
        efficiency = (self.optimal_steps / avg_steps * 100) if avg_steps > 0 else 0
        
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'num_episodes': num_episodes,
            'evaluation_time': evaluation_time,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward,
            'avg_episode_time': avg_episode_time,
            'optimal_solves': results['optimal_solves'],
            'near_optimal': results['near_optimal'],
            'efficiency_score': efficiency,
            'step_distribution': {
                'min': min(results['step_distribution']) if results['step_distribution'] else 0,
                'max': max(results['step_distribution']) if results['step_distribution'] else 0,
                'median': np.median(results['step_distribution']) if results['step_distribution'] else 0,
                'std': np.std(results['step_distribution']) if results['step_distribution'] else 0
            }
        }
        
        self.evaluation_history.append(evaluation_report)
        
        return evaluation_report
    
    def generate_report(self, report, model_name="Current Model"):
        """Generate a human-readable report"""
        print("\n" + "="*80)
        print(f"MODEL EVALUATION REPORT - {model_name}")
        print("="*80)
        print(f"Evaluation Date: {report['timestamp']}")
        print(f"Evaluation Time: {report['evaluation_time']:.2f}s")
        print(f"Episodes Tested: {report['num_episodes']}")
        print()
        print("PERFORMANCE METRICS")
        print("-"*80)
        print(f"Success Rate:        {report['success_rate']:.2f}%")
        print(f"Average Steps:       {report['avg_steps']:.2f} (Optimal: {self.optimal_steps})")
        print(f"Efficiency Score:    {report['efficiency_score']:.2f}%")
        print(f"Average Reward:      {report['avg_reward']:.2f}")
        print()
        print("SOLUTION QUALITY")
        print("-"*80)
        print(f"Optimal Solves:      {report['optimal_solves']} ({report['optimal_solves']/report['num_episodes']*100:.1f}%)")
        print(f"Near-Optimal:        {report['near_optimal']} ({report['near_optimal']/report['num_episodes']*100:.1f}%)")
        print()
        print("STEP DISTRIBUTION")
        print("-"*80)
        print(f"Minimum Steps:       {report['step_distribution']['min']}")
        print(f"Median Steps:        {report['step_distribution']['median']:.1f}")
        print(f"Maximum Steps:       {report['step_distribution']['max']}")
        print(f"Std Deviation:       {report['step_distribution']['std']:.2f}")
        print()
        print("SPEED")
        print("-"*80)
        print(f"Avg Time/Episode:    {report['avg_episode_time']*1000:.2f}ms")
        print("="*80 + "\n")
        
        return report
    
    def compare_models(self, reports, model_names=None):
        """Compare multiple model evaluation reports"""
        if not reports:
            print("No reports to compare")
            return
        
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(reports))]
        
        print("\n" + "="*100)
        print("MODEL COMPARISON REPORT")
        print("="*100)
        print(f"{'Metric':<30} | {' | '.join(f'{name:^20}' for name in model_names)}")
        print("-"*100)
        
        metrics = [
            ('Success Rate (%)', 'success_rate', '{:.2f}'),
            ('Avg Steps', 'avg_steps', '{:.2f}'),
            ('Efficiency Score (%)', 'efficiency_score', '{:.2f}'),
            ('Optimal Solves', 'optimal_solves', '{}'),
            ('Near-Optimal', 'near_optimal', '{}'),
            ('Avg Reward', 'avg_reward', '{:.2f}'),
            ('Median Steps', lambda r: r['step_distribution']['median'], '{:.1f}'),
            ('Evaluation Time (s)', 'evaluation_time', '{:.2f}')
        ]
        
        for metric_name, metric_key, fmt in metrics:
            values = []
            for report in reports:
                if callable(metric_key):
                    value = metric_key(report)
                else:
                    value = report[metric_key]
                values.append(value)
            
            value_strs = [fmt.format(v) for v in values]
            
            # Highlight best value
            if metric_name in ['Success Rate (%)', 'Efficiency Score (%)', 'Optimal Solves', 'Near-Optimal', 'Avg Reward']:
                best_idx = values.index(max(values))
            else:
                best_idx = values.index(min(values))
            
            value_strs[best_idx] = f"★ {value_strs[best_idx]}"
            
            print(f"{metric_name:<30} | {' | '.join(f'{v:^20}' for v in value_strs)}")
        
        print("="*100 + "\n")
        
        # Determine overall winner
        scores = []
        for report in reports:
            score = (
                report['success_rate'] * 0.4 +  # 40% weight on success
                report['efficiency_score'] * 0.3 +  # 30% weight on efficiency
                (report['optimal_solves'] / report['num_episodes'] * 100) * 0.3  # 30% on optimal
            )
            scores.append(score)
        
        winner_idx = scores.index(max(scores))
        print(f"🏆 OVERALL WINNER: {model_names[winner_idx]} (Score: {scores[winner_idx]:.2f})")
        print()
        
    def save_report(self, report, filename):
        """Save evaluation report to JSON file"""
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {filepath}")
    
    def load_report(self, filename):
        """Load evaluation report from JSON file"""
        with open(filename, 'r') as f:
            report = json.load(f)
        return report
    
    def track_learning_progress(self, metrics_obj, episodes_checkpoint):
        """
        Track how quickly the model is learning
        Returns learning rate metrics
        """
        history = metrics_obj.history
        
        if len(history) < 100:
            return None
        
        # Calculate learning rate over different windows
        windows = [100, 500, 1000]
        learning_metrics = {
            'episodes_trained': len(history),
            'learning_windows': {}
        }
        
        for window in windows:
            if len(history) >= window:
                recent = history[-window:]
                successes = sum(1 for ep in recent if ep['success'])
                success_rate = (successes / len(recent)) * 100
                avg_steps = np.mean([ep['steps'] for ep in recent if ep['success']])
                
                learning_metrics['learning_windows'][f'last_{window}'] = {
                    'success_rate': success_rate,
                    'avg_steps': avg_steps
                }
        
        # Calculate convergence: episodes needed to reach 80% success rate
        convergence_episode = None
        for i in range(100, len(history)):
            window = history[i-100:i]
            successes = sum(1 for ep in window if ep['success'])
            if successes >= 80:  # 80% of 100 episodes
                convergence_episode = i
                break
        
        learning_metrics['convergence_episode'] = convergence_episode
        learning_metrics['convergence_time'] = f"{convergence_episode} episodes" if convergence_episode else "Not yet converged"
        
        return learning_metrics


class LearningRateTracker:
    """Tracks and compares learning rates across different training sessions"""
    
    def __init__(self, save_dir="models/learning_reports"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save_training_session(self, session_data):
        """Save a complete training session report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.save_dir / f"training_session_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return filename
    
    def load_all_sessions(self):
        """Load all training session reports"""
        sessions = []
        for file in sorted(self.save_dir.glob("training_session_*.json")):
            with open(file, 'r') as f:
                sessions.append(json.load(f))
        return sessions
    
    def compare_learning_rates(self):
        """Compare learning rates across all sessions"""
        sessions = self.load_all_sessions()
        
        if not sessions:
            print("No training sessions found to compare")
            return
        
        print("\n" + "="*100)
        print("LEARNING RATE COMPARISON")
        print("="*100)
        
        for i, session in enumerate(sessions, 1):
            print(f"\nSession {i}: {session.get('session_name', 'Unnamed')}")
            print(f"  Date: {session.get('timestamp', 'Unknown')}")
            print(f"  Total Episodes: {session.get('total_episodes', 'N/A')}")
            print(f"  Training Time: {session.get('total_training_time', 0):.2f}s")
            
            if 'final_evaluation' in session:
                eval_data = session['final_evaluation']
                print(f"  Final Success Rate: {eval_data.get('success_rate', 0):.2f}%")
                print(f"  Final Efficiency: {eval_data.get('efficiency_score', 0):.2f}%")
            
            if 'convergence_episode' in session:
                conv = session['convergence_episode']
                print(f"  Convergence: {conv if conv else 'Not achieved'}")
        
        print("="*100 + "\n")
