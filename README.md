# Tower of Hanoi - Reinforcement Learning with PyQt6 GUI

A comprehensive reinforcement learning solution for solving the Tower of Hanoi puzzle with beautiful animated PyQt6 GUI, interactive menu launcher, comprehensive model evaluation, and learning rate tracking.

## Features

### 🖥️ Modern GUI Interface
- **Interactive Menu**: Launch all features from a beautiful PyQt6 GUI
- **Training Configuration Dialog**: Easy parameter setup with spinners and validation
- **Model Selection**: Visual list of trained models with timestamps
- **Model Comparison**: Multi-select interface for comparing multiple models
- **Learning Reports**: View training history and session comparisons
- **Tutorial Dialog**: Rich HTML-formatted educational content
- **Progress Feedback**: Real-time status updates and result displays

### 🎮 Advanced Visualization
- **Smooth Disc Animations**: 30-frame animations showing lift, move, and drop phases
- **Thread-safe Rendering**: QTimer-based animation system prevents UI freezes
- **Visualization Toggle**: Switch between animated and fast training modes
- **Real-time Metrics**: Always-updated episode count, success rate, and epsilon
- **Color-coded Discs**: Each disc has a unique color for easy tracking
- **Configurable Speed**: Adjust animation timing with slider control

### 🤖 Deep Q-Learning (DQN)
- **Neural Network Agent**: 24-24 hidden layer architecture
- **Experience Replay**: 2000-memory buffer for stable training
- **Epsilon-Greedy Exploration**: Balanced exploration vs exploitation (0.95 → 0.01)
- **Adaptive Learning**: Epsilon decay for progressive strategy refinement
- **Reward System**: +100 success, -1 per move, -10 invalid moves

### 📊 Comprehensive Model Evaluation
- **Performance Metrics**: Success rate, average steps, efficiency score, optimal solves
- **Learning Progress Tracking**: Analyze convergence speed and learning windows
- **Model Comparison**: Side-by-side comparison with weighted scoring system
- **Session History**: Track all training sessions with timestamps and metrics
- **Convergence Detection**: Identifies episode when 80% success achieved
- **Automated Reporting**: JSON reports for all evaluations and training sessions

### 🔄 Intelligent Model Management
- **Automatic Checkpointing**: Save models at regular intervals
- **Timestamped Models**: Organized with creation date/time in filename
- **Performance Tracking**: Metadata for each saved model
- **Test Anytime**: Load and test any saved model with visualization
- **Standalone Testing**: CLI tool for batch model evaluation

## Installation

1. **Clone or navigate to the project directory:**
```bash
cd /Users/loganathan.sekaran/git/toh
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.12+
- TensorFlow 2.20.0+
- NumPy
- PyQt6 6.10.0+ (replaces tkinter)
- Gymnasium 1.2.1+ (modern replacement for OpenAI Gym)

## Quick Start

### Launch GUI (Default)
```bash
./start.sh
# or
python main.py
```

The modern PyQt6 GUI will launch with buttons for:
- 🎬 **Demo** - Watch optimal solution
- 🏋️ **Train Model** - Configure and train new model
- 🧪 **Test Model** - Test saved models with visualization
- ⚡ **Quick Train** - Fast 500-episode training
- 📊 **Compare Models** - Side-by-side model comparison
- 📈 **Learning Reports** - View training history
- 📚 **Tutorial** - Learn how it works
- ❌ **Exit**

### Alternative Launch Methods
```bash
# Direct GUI launch
./start_gui.sh
python gui_launcher.py

# CLI menu (legacy)
python main.py --cli

# Direct commands
python main.py demo
python main.py train --episodes 2000
python main.py test --model models/my_model.weights.h5
```

## GUI Features

### Training Configuration Dialog
When you click **Train Model**, a dialog appears to configure:
- **Number of Discs**: 3-5 (default: 3)
- **Training Episodes**: 100-10,000 (default: 1000)
- **Batch Size**: 16-128 (default: 32)

### Model Selection
Visual list showing all trained models with:
- Model filename
- Creation timestamp
- Most recent models first

### Model Comparison
Multi-select interface (Ctrl+Click) to compare models:
- Select 2+ models
- Click "Compare Selected Models"
- View detailed comparison table
- See winner with weighted scoring

### Learning Reports
Automatically displays:
- All training sessions with timestamps
- Episodes trained and duration
- Convergence metrics
- Success rates and efficiency scores
- Cross-session comparisons

### Tutorial
Rich HTML-formatted content explaining:
- Tower of Hanoi rules
- Optimal solution math (2^N - 1)
- How DQN learning works
- Training process overview
- Tips for best results

## Quick Start

### 1. Demo the Visualizer (Optimal Solution)
See the visualizer in action with the optimal algorithm:
```bash
python main.py demo
```

For more discs:
```bash
python main.py demo --discs 4
```

### 2. Train a Model
Train a new RL agent with visualization:
```bash
python main.py train
```

**Training Options:**
```bash
# Train with custom settings
python main.py train --discs 3 --episodes 1000 --speed 0.3 --show-every 10

# Quick training with more visualization
python main.py train --episodes 500 --show-every 5 --speed 0.2

# Challenge: Train with 4 discs
python main.py train --discs 4 --episodes 2000
```

**Parameters:**
- `--discs`: Number of discs (default: 3)
- `--episodes`: Total training episodes (default: 1000)
- `--speed`: Animation speed in seconds (default: 0.3)
- `--show-every`: Show visualization every N episodes (default: 10)

### 3. Test a Trained Model
Evaluate your trained model:
```bash
python main.py test
```

**Test Options:**
```bash
# Test specific model
python main.py test --model models/model_v5.h5

# Test with slower animation
python main.py test --speed 1.0
```

## Project Structure

```
toh/
├── main.py                 # Main launcher script
├── toh.py                  # Tower of Hanoi environment
├── dqn_agent.py            # DQN reinforcement learning agent
├── visualizer.py           # Tkinter-based GUI visualizer
├── train_with_gui.py       # Training script with visualization
├── util.py                 # Utility functions
├── requirements.txt        # Python dependencies
├── models/                 # Saved model checkpoints
│   ├── model_v*.h5        # Model files
│   └── model_v*_metadata.json  # Model metadata
└── training_metrics.json   # Training history
```

## How It Works

### Environment (toh.py)
- **State**: 3 rods with discs represented as lists
- **Actions**: 6 possible moves (rod i → rod j, i ≠ j)
- **Rewards**:
  - -1 for each step
  - -10 for invalid moves
  - +100 for completing the puzzle
- **Goal**: Move all discs from rod 1 to rod 3

### Agent (dqn_agent.py)
- **Architecture**: 2-layer neural network (24-24 nodes)
- **Input**: Flattened state (9 values for 3 discs)
- **Output**: Q-values for 6 possible actions
- **Training**: Experience replay with batch size 64
- **Exploration**: Epsilon-greedy (ε starts at 1.0, decays to 0.01)

### Visualizer (visualizer.py)
- **Framework**: Tkinter for cross-platform GUI
- **Animation**: Smooth disc movement with configurable speed
- **Metrics Display**: Real-time training statistics
- **Controls**: Interactive speed adjustment

### Model Management (train_with_gui.py)
- **Checkpointing**: Save every 100 episodes
- **Best Model Tracking**: Keep best model based on success rate
- **Auto-replacement**: Replace failing models after 3 evaluations
- **Performance Criteria**:
  - Primary: Success rate
  - Secondary: Average steps (for successful episodes)

## Training Details

### Optimal Solution
For n discs, the optimal solution requires **2^n - 1** moves:
- 3 discs: 7 moves
- 4 discs: 15 moves
- 5 discs: 31 moves

### Expected Training Progress
- **Episodes 1-100**: Random exploration, low success rate (~0-10%)
- **Episodes 100-300**: Learning patterns, improving success rate (~10-50%)
- **Episodes 300-500**: Consistent success, optimizing moves (~50-80%)
- **Episodes 500+**: High success rate, near-optimal moves (~80-95%)

### Performance Metrics
Monitor these during training:
- **Success Rate**: % of episodes that successfully complete the puzzle
- **Average Steps**: Mean steps for successful episodes (compare to optimal)
- **Epsilon**: Exploration rate (should decrease over time)
- **Total Reward**: Higher is better (should increase over time)

## Advanced Usage

### Custom Training Loop
You can create your own training script:

```python
from toh import TowerOfHanoiEnv
from dqn_agent import DQNAgent
from visualizer import TowerOfHanoiVisualizer
from train_with_gui import train_with_visualization, ModelManager
import numpy as np

# Setup
env = TowerOfHanoiEnv(num_discs=3)
agent = DQNAgent(state_size=9, action_size=6)
visualizer = TowerOfHanoiVisualizer(num_discs=3, animation_speed=0.3)
model_manager = ModelManager()

# Train
metrics = train_with_visualization(
    env, agent, visualizer, num_discs=3,
    total_episodes=1000,
    model_manager=model_manager
)

# Keep window open
visualizer.root.mainloop()
```

### Analyzing Training Metrics
Training metrics are saved to `training_metrics.json`:

```python
import json
import matplotlib.pyplot as plt

# Load metrics
with open('training_metrics.json', 'r') as f:
    metrics = json.load(f)

# Plot success rate over time
successes = metrics['successes']
window = 100
success_rate = [np.mean(successes[max(0, i-window):i+1]) * 100 
                for i in range(len(successes))]

plt.plot(success_rate)
plt.xlabel('Episode')
plt.ylabel('Success Rate (%)')
plt.title('Training Progress')
plt.show()
```

## Troubleshooting

### Training is too slow
- Increase `--show-every` to reduce visualization frequency
- Reduce `--speed` for faster animations (minimum 0.1s)
- Train without visualization by modifying `train_with_gui.py`

### Model not learning
- Ensure enough training episodes (try 1000+)
- Check if rewards are increasing over time
- Monitor epsilon decay (should decrease gradually)
- Try adjusting hyperparameters in `dqn_agent.py`

### Visualization window not responding
- Reduce animation speed
- Ensure tkinter is properly installed
- Check system resources

## Configuration

### Hyperparameters (dqn_agent.py)
```python
gamma = 0.95           # Discount factor
epsilon = 1.0          # Initial exploration rate
epsilon_min = 0.01     # Minimum exploration rate
epsilon_decay = 0.995  # Exploration decay rate
learning_rate = 0.001  # Neural network learning rate
batch_size = 64        # Replay batch size
memory_size = 2000     # Experience replay buffer size
```

### Training Configuration (train_with_gui.py)
```python
eval_interval = 50        # Evaluate every N episodes
checkpoint_interval = 100 # Save checkpoint every N episodes
show_every_n = 10         # Visualize every N episodes
```

## Tips for Best Results

1. **Start with 3 discs** to ensure the agent learns the basics
2. **Monitor success rate** - should reach >80% after 500-1000 episodes
3. **Let it train longer** - more episodes = better performance
4. **Use model replacement** - automatically handles performance degradation
5. **Experiment with hyperparameters** - adjust learning rate, epsilon decay, etc.

## Future Enhancements

Potential improvements:
- [ ] Add more visualization options (graphs, charts)
- [ ] Implement A3C or PPO algorithms
- [ ] Add curriculum learning (start with 2 discs, increase gradually)
- [ ] Support for more than 3 rods
- [ ] Real-time hyperparameter tuning
- [ ] Tensorboard integration
- [ ] Multi-agent training

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Acknowledgments

- Farama Foundation for Gymnasium (successor to OpenAI Gym)
- TensorFlow team for the deep learning framework
- Tower of Hanoi puzzle - a classic in computer science education
