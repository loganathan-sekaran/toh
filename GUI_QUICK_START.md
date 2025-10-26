# GUI Launch Guide

## Quick Start

Run the application:
```bash
./start.sh
```

The GUI will launch automatically with this menu:

```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║       🗼  Tower of Hanoi - RL Trainer  🗼                 ║
║                                                            ║
║   Train an AI agent to solve Tower of Hanoi using DQN    ║
║                                                            ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  ┌──────────────────────────────────────────────────┐    ║
║  │  🎬  Demo                                         │    ║
║  │  Watch the optimal solution with animation       │    ║
║  └──────────────────────────────────────────────────┘    ║
║                                                            ║
║  ┌──────────────────────────────────────────────────┐    ║
║  │  🏋️  Train Model                                  │    ║
║  │  Train a new AI model with visualization         │    ║
║  └──────────────────────────────────────────────────┘    ║
║                                                            ║
║  ┌──────────────────────────────────────────────────┐    ║
║  │  🧪  Test Model                                   │    ║
║  │  Test a trained model with visualization         │    ║
║  └──────────────────────────────────────────────────┘    ║
║                                                            ║
║  ┌──────────────────────────────────────────────────┐    ║
║  │  ⚡  Quick Train                                   │    ║
║  │  Fast training (500 episodes, no visualization)  │    ║
║  └──────────────────────────────────────────────────┘    ║
║                                                            ║
║  ┌──────────────────────────────────────────────────┐    ║
║  │  📊  Compare Models                               │    ║
║  │  Compare performance of multiple models          │    ║
║  └──────────────────────────────────────────────────┘    ║
║                                                            ║
║  ┌──────────────────────────────────────────────────┐    ║
║  │  📈  Learning Reports                             │    ║
║  │  View training session history and learning rates│    ║
║  └──────────────────────────────────────────────────┘    ║
║                                                            ║
║  ┌──────────────────────────────────────────────────┐    ║
║  │  📚  Tutorial                                     │    ║
║  │  Learn how Tower of Hanoi works                  │    ║
║  └──────────────────────────────────────────────────┘    ║
║                                                            ║
║  ┌──────────────────────────────────────────────────┐    ║
║  │  ❌  Exit                                          │    ║
║  └──────────────────────────────────────────────────┘    ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

## Button Actions

### 🎬 Demo
**Click** → Shows optimal 7-move solution with animation

### 🏋️ Train Model
**Click** → Opens dialog:
```
┌─ Configure Training ──────────┐
│                                │
│  Number of Discs:    [3 ▼]   │
│  Training Episodes:  [1000 ▼] │
│  Batch Size:        [32 ▼]   │
│                                │
│  💡 Tip: More episodes = better│
│     learning but longer time   │
│                                │
│  [  OK  ]  [ Cancel ]         │
└────────────────────────────────┘
```
**Click OK** → Starts training with visualization

### 🧪 Test Model
**Click** → Opens model selection:
```
┌─ Select Model ────────────────────┐
│                                    │
│  Select a model to test:           │
│                                    │
│  ┌────────────────────────────┐  │
│  │ toh_3discs_20250121.h5     │  │
│  │   Created: 2025-01-21 14:30│  │
│  │                             │  │
│  │ toh_3discs_20250120.h5     │  │
│  │   Created: 2025-01-20 16:45│  │
│  └────────────────────────────┘  │
│                                    │
│  [  OK  ]  [ Cancel ]             │
└────────────────────────────────────┘
```
**Select & OK** → Tests model with visualization

### ⚡ Quick Train
**Click** → Shows confirmation:
```
┌─ Quick Train ─────────────────────┐
│                                    │
│  Start fast training with 500     │
│  episodes?                         │
│                                    │
│  No visualization will be shown    │
│  during training.                  │
│                                    │
│  [  Yes  ]  [  No  ]              │
└────────────────────────────────────┘
```
**Click Yes** → Trains quickly in background

### 📊 Compare Models
**Click** → Opens comparison dialog:
```
┌─ Compare Models ──────────────────┐
│                                    │
│  Select 2+ models (Ctrl+Click):   │
│                                    │
│  ┌────────────────────────────┐  │
│  │ ☑ toh_3discs_20250121.h5   │  │
│  │ ☑ toh_3discs_20250120.h5   │  │
│  │ ☐ toh_3discs_20250119.h5   │  │
│  └────────────────────────────┘  │
│                                    │
│  [ 📊 Compare Selected Models ]   │
│                                    │
│  Results:                          │
│  ┌────────────────────────────┐  │
│  │ Model Comparison Report     │  │
│  │ ========================    │  │
│  │ Model 1: 95% success       │  │
│  │ Model 2: 88% success       │  │
│  │ WINNER: Model 1            │  │
│  └────────────────────────────┘  │
│                                    │
│  [ Close ]                         │
└────────────────────────────────────┘
```

### 📈 Learning Reports
**Click** → Shows training history:
```
┌─ Learning Reports ────────────────┐
│                                    │
│  Training Session History:         │
│                                    │
│  ┌────────────────────────────┐  │
│  │ Session 1                   │  │
│  │   Date: 2025-01-21 14:30   │  │
│  │   Episodes: 1000            │  │
│  │   Success: 95%              │  │
│  │   Convergence: Ep 287      │  │
│  │                             │  │
│  │ Session 2                   │  │
│  │   Date: 2025-01-20 16:45   │  │
│  │   Episodes: 500             │  │
│  │   Success: 88%              │  │
│  │   Convergence: Ep 412      │  │
│  └────────────────────────────┘  │
│                                    │
│  [ Close ]                         │
└────────────────────────────────────┘
```

### 📚 Tutorial
**Click** → Shows tutorial:
```
┌─ Tower of Hanoi Tutorial ─────────┐
│                                    │
│  🗼 Tower of Hanoi - How It Works │
│                                    │
│  ┌────────────────────────────┐  │
│  │ 📖 The Rules                │  │
│  │ • 3 rods (A, B, C)          │  │
│  │ • N discs of different sizes│  │
│  │ • Move all to rod C         │  │
│  │ • Only smaller on larger    │  │
│  │                             │  │
│  │ 🎯 Optimal Solution         │  │
│  │ • Formula: 2^N - 1 moves    │  │
│  │ • 3 discs = 7 moves         │  │
│  │                             │  │
│  │ 🤖 How the AI Learns        │  │
│  │ • Deep Q-Learning (DQN)     │  │
│  │ • Neural network learns     │  │
│  │   best actions              │  │
│  │ ...                         │  │
│  └────────────────────────────┘  │
│                                    │
│  [ Close ]                         │
└────────────────────────────────────┘
```

## Alternative Launch Methods

### Direct GUI Launch
```bash
./start_gui.sh
# or
python gui_launcher.py
```

### Force CLI Menu (Old Style)
```bash
python main.py --cli
# or
python main.py cli
```

### Direct Commands (Skip Menu)
```bash
python main.py demo
python main.py train --episodes 2000
python main.py test --model models/my_model.weights.h5
```

## First-Time User Workflow

1. **Launch**: Run `./start.sh`
2. **Learn**: Click **📚 Tutorial** to understand the game
3. **Demo**: Click **🎬 Demo** to see optimal solution
4. **Train**: Click **🏋️ Train Model**, use defaults (1000 episodes)
5. **Test**: Click **🧪 Test Model**, select your trained model
6. **Compare**: Train 2-3 more models, then click **📊 Compare Models**

## Tips

- **Fast Training**: Use **⚡ Quick Train** or toggle visualization during regular training
- **Model Organization**: Models save automatically with timestamps
- **Progress Tracking**: Use **📈 Learning Reports** to see improvement over time
- **Best Results**: Train 3-5 models and compare to find optimal settings

## See Also

- `GUI_DOCUMENTATION.md` - Complete GUI reference
- `MODEL_EVALUATION.md` - Understanding metrics
- `ANIMATION_FIXES.md` - Technical details

---

**Enjoy training your AI! 🎯**
