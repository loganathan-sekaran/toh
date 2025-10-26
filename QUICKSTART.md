# Tower of Hanoi RL - Quick Reference

## ✅ All Systems Operational

### Start the Application
```bash
./start.sh
```

### Or Use Python Directly
```bash
# Interactive menu
python main.py

# Demo mode - watch optimal solution
python main.py demo

# Train new model
python main.py train --episodes 1000 --discs 3

# Test trained model
python main.py test --model models/best_model.weights.h5
```

### Verify Installation
```bash
python test_setup.py
```

### Test Training (No GUI)
```bash
python test_training.py
```

## 📁 Key Files

- **main.py** - Main launcher with interactive menu
- **train_with_gui.py** - Training with PyQt6 visualization
- **visualizer.py** - PyQt6 GUI with animations
- **toh.py** - Tower of Hanoi environment
- **dqn_agent.py** - Deep Q-Network agent
- **util.py** - State transformation utilities

## 🎯 Success Criteria

- Success rate > 80%
- Average steps approaching 7 (optimal for 3 discs)
- Epsilon decay from 1.0 → 0.01
- Consistent puzzle solving

## 🐛 Troubleshooting

**GUI not showing?**
```bash
# Check PyQt6
python -c "from PyQt6.QtWidgets import QApplication; print('✓ Working')"
```

**Import errors?**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Training not working?**
```bash
# Run test suite
python test_setup.py
python test_training.py
```

## 💡 Tips

1. Start with demo mode to see optimal solution
2. Train for 500-1000 episodes initially
3. Use speed slider to slow down animations
4. Pause training to inspect agent behavior
5. Models auto-save when improving

---

**System ready! Start with: `./start.sh`**
