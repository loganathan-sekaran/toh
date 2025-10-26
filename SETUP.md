# Setup Complete! 🎉

Your Tower of Hanoi Reinforcement Learning environment is ready to use!

## Virtual Environment

A Python virtual environment has been created at `venv/` with all dependencies installed:
- ✅ Python 3.10.15 (optimized for TensorFlow GPU support and tkinter)
- ✅ TensorFlow 2.20.0
- ✅ NumPy 2.2.6
- ✅ Matplotlib 3.10.7
- ✅ Gymnasium 1.2.1 (modern replacement for deprecated Gym)
- ✅ tkinter (for interactive GUI with controls)

## Quick Start

### Option 1: Use the Interactive Launcher (Recommended)
```bash
./start.sh
```
This will activate the venv and launch the interactive menu.

### Option 2: Activate venv and run directly
```bash
source venv/bin/activate
python quickstart.py
```

### Option 3: Run specific commands
```bash
source venv/bin/activate

# Demo the visualizer
python main.py demo

# Train a model
python main.py train

# Test a model
python main.py test
```

## What to Do Next

1. **First Time?** Start with the demo:
   ```bash
   source venv/bin/activate
   python main.py demo
   ```

2. **Ready to Train?** Train your first model:
   ```bash
   source venv/bin/activate
   python main.py train --episodes 500 --show-every 5
   ```

3. **Test Your Model:**
   ```bash
   source venv/bin/activate
   python main.py test
   ```

## Understanding the Files

- `main.py` - Command-line interface for training/testing
- `quickstart.py` - Interactive menu system
- `train_with_gui.py` - Training with GUI visualization
- `visualizer.py` - Tkinter-based animated visualization
- `toh.py` - Tower of Hanoi environment
- `dqn_agent.py` - Deep Q-Network agent
- `models/` - Saved model checkpoints (created during training)
- `training_metrics.json` - Training statistics (created during training)

## Deactivating the Virtual Environment

When you're done:
```bash
deactivate
```

## Troubleshooting

### If tkinter is not available:
On macOS, tkinter comes with Python. If you have issues:
```bash
brew install python-tk
```

### If you need to reinstall dependencies:
```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### If you want to start fresh:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Need Help?

Read the full documentation in `README.md`:
```bash
cat README.md
```

Or just run the interactive launcher and choose option 5 (Tutorial):
```bash
./start.sh
```

---

**Ready to train your AI? Let's go!** 🚀
