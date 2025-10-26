# Tower of Hanoi - Reinforcement Learning with PyQt6 GUI

Successfully migrated from tkinter to **PyQt6** for better cross-platform compatibility and zero dependency issues!

## ✨ Features

- **Beautiful PyQt6 GUI** with smooth animations
- **Deep Q-Network (DQN)** reinforcement learning
- **Interactive Controls**: Pause/Resume, Stop, Speed slider
- **Real-time Metrics**: Episode count, steps, rewards, epsilon, success rate
- **Model Management**: Automatic checkpointing and best model selection
- **Multiple Modes**: Demo, Train, Test

## 🚀 Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run interactive launcher
./start.sh

# Or use Python directly
python main.py
```

## 📋 Menu Options

1. **🎬 Demo** - Watch the optimal solution (7 moves for 3 discs)
2. **🏋️ Train** - Train a new AI model with visualization
3. **🧪 Test** - Test a trained model
4. **⚡ Quick Train** - Fast training (500 episodes)
5. **🎓 Tutorial** - Learn about Tower of Hanoi
6. **❌ Exit**

## 💻 Command Line Usage

```bash
# Demo mode
python main.py demo

# Training
python main.py train --episodes 1000 --discs 3

# Testing
python main.py test --model models/best_model.weights.h5 --discs 3
```

## 🎯 Why PyQt6?

✅ **Zero dependency issues** - Installs cleanly via pip  
✅ **Cross-platform** - Works on macOS, Linux, Windows  
✅ **Modern UI** - Beautiful anti-aliased graphics  
✅ **Rich controls** - Better widgets and styling  
✅ **Stable** - No platform-specific build issues  

## 📦 Dependencies

```
gymnasium       # Modern RL environment
tensorflow      # Deep learning
PyQt6          # GUI framework (replaces tkinter)
matplotlib     # Plotting
numpy          # Numerical operations
```

## 🎮 GUI Controls

- **Pause/Resume** - Pause training to inspect current state
- **Stop Training** - Stop and save progress
- **Speed Slider** - Adjust animation speed (1-100 ms)
- **Test Model** - Evaluate trained models

## 📊 Metrics Tracked

- Episode number
- Steps per episode
- Total reward
- Epsilon (exploration rate)
- Success rate (last 100 episodes)
- Average steps (successful episodes)

## 🏗️ Project Structure

```
toh/
├── main.py                  # Interactive launcher
├── train_with_gui.py       # Training with PyQt6 GUI
├── visualizer.py           # PyQt6 visualizer
├── toh.py                  # Tower of Hanoi environment
├── dqn_agent.py           # DQN agent implementation
├── util.py                # Helper functions
├── test_setup.py          # Verification script
├── requirements.txt       # Dependencies
└── models/                # Saved models
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_setup.py
```

## 🎓 How It Works

The agent uses **Deep Q-Learning (DQN)** to learn optimal Tower of Hanoi strategies:

1. **State Representation**: 3x3 matrix (3 rods × 3 disc positions)
2. **Actions**: 6 possible moves (rod A→B, A→C, B→A, B→C, C→A, C→B)
3. **Rewards**:
   - +100: Successfully solving the puzzle
   - -1: Each move (encourages efficiency)
   - -10: Invalid move (encourages valid play)
4. **Neural Network**: 2 hidden layers (24 neurons each)
5. **Experience Replay**: Learns from past experiences
6. **Epsilon-Greedy**: Balances exploration vs exploitation

## 📈 Training Tips

- Start with 3 discs (optimal solution: 7 moves)
- Train for 1000+ episodes
- Success rate should reach >80% for good models
- Average steps should approach 7 (optimal)
- Epsilon decay helps transition from exploration to exploitation

## 🔧 Troubleshooting

**ImportError: No module named 'PyQt6'**
```bash
pip install PyQt6
```

**GUI not showing**
- Ensure you're not in a headless environment
- Check that PyQt6 installed correctly: `python -c "from PyQt6.QtWidgets import QApplication; print('✓ Working')"`

## 📝 License

Educational project for demonstrating reinforcement learning with PyQt6 GUI.

## 🙏 Acknowledgments

- OpenAI Gymnasium for the RL framework
- PyQt6 for the excellent GUI toolkit
- TensorFlow/Keras for deep learning

---

**Happy Training! 🎉**
