# Modular Model Architecture System

## Overview

The Tower of Hanoi RL Trainer now features a **modular model architecture system** that allows you to:

✅ Define different neural network architectures in separate files  
✅ Automatically discover and register new architectures  
✅ Select architectures via GUI dropdown during training  
✅ Compare performance across different models  
✅ Track which architecture was used for each trained model  

## Architecture Files

All model architectures are located in `model_architectures/` directory:

```
model_architectures/
├── __init__.py              # Package initialization
├── base_architecture.py     # Abstract base class
├── model_factory.py         # Auto-discovery and registration
├── small_model.py           # Small 24-24 architecture
├── medium_model.py          # Medium 64-32 architecture
├── large_model.py           # Large 128-64-32 architecture (default)
└── extra_large_model.py     # Extra Large 256-128-64 architecture
```

## Available Architectures

| Architecture | Layers | Parameters | Complexity | Use Case |
|-------------|--------|------------|------------|----------|
| **Small (24-24)** | 24→24 | 990 | Low | Fast experiments, baseline |
| **Medium (64-32)** | 64→32 | 2,918 | Medium | Balanced performance/speed |
| **Large (128-64-32)** | 128→64→32 | 11,814 | High | **Best results (default)** |
| **Extra Large (256-128-64)** | 256→128→64 | 44,102 | Very High | Maximum capacity |

## How to Use

### 1. Via GUI (Easiest)

```bash
./start_gui.sh
```

1. Click **"🏋️ Train Model"**
2. Select **Model Architecture** from dropdown
3. The description and recommended episodes auto-update
4. Configure other parameters
5. Click OK to train

### 2. Via Code

```python
from dqn_agent import DQNAgent

# Create agent with specific architecture
agent = DQNAgent(
    state_size=9,
    action_size=6,
    architecture_name="Large (128-64-32)"
)

# Available architectures:
# - "Small (24-24)"
# - "Medium (64-32)"
# - "Large (128-64-32)"
# - "Extra Large (256-128-64)"
```

### 3. List Available Architectures

```python
from model_architectures import ModelFactory

# List all architectures
ModelFactory.list_architectures()

# Get architecture names
names = ModelFactory.get_architecture_names()

# Get specific architecture info
info = ModelFactory.get_architecture_info("Large (128-64-32)")
print(info['description'])
print(info['recommended_episodes'])
```

## Creating Your Own Architecture

Want to experiment with a new architecture? Easy!

### Step 1: Create New File

Create `model_architectures/my_custom_model.py`:

```python
"""
My Custom Model Architecture
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from .base_architecture import ModelArchitecture


class MyCustomModel(ModelArchitecture):
    """
    Description of your model
    """
    
    def __init__(self):
        super().__init__()
        self.name = "My Custom (512-256-128)"
        self.description = "Ultra deep model with 3 large hidden layers"
        self.recommended_episodes = 2500
        self.complexity = "Very High"
    
    def build(self, state_size: int, action_size: int, learning_rate: float) -> Sequential:
        """Build your custom model."""
        model = Sequential([
            Dense(512, input_dim=state_size, activation='relu', 
                  kernel_initializer='he_uniform'),
            Dense(256, activation='relu', kernel_initializer='he_uniform'),
            Dense(128, activation='relu', kernel_initializer='he_uniform'),
            Dense(action_size, activation='linear', 
                  kernel_initializer='he_uniform')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='huber',
            metrics=['mae']
        )
        
        return model
```

### Step 2: That's It! 🎉

Your architecture is **automatically discovered** and will appear in:
- GUI dropdown menu
- ModelFactory.list_architectures()
- ModelFactory.get_architecture_names()

No registration code needed!

## Model Metadata

When you train a model, the following metadata is automatically saved:

```json
{
  "name": "model_20251026_143022",
  "architecture": "Large (128-64-32)",
  "created_at": "2025-10-26T14:30:22",
  "state_size": 9,
  "action_size": 6,
  "episodes": 1500,
  "success_rate": 95.3,
  "avg_steps": 8.2,
  "epsilon": 0.015,
  "num_discs": 3
}
```

## Testing Different Models

### Compare Performance

1. Train multiple models with different architectures
2. Click **"📊 Compare Models"** in GUI
3. View performance metrics side-by-side:
   - Architecture used
   - Success rate
   - Average steps
   - Training episodes

### Find Optimal Architecture

**Goal**: Find architecture that learns fastest to optimal 7-step solution

1. Train each architecture (500-2000 episodes)
2. Compare results:
   - **Success Rate**: Should be >90%
   - **Avg Steps**: Closer to 7 is better
   - **Training Time**: Smaller models train faster

**Recommendation**: Start with **Large (128-64-32)** - best balance of performance and speed.

## Architecture Design Guidelines

When creating custom architectures:

### 1. Layer Size Progression

**Good**: Funnel pattern (decreasing)
```python
Dense(128) → Dense(64) → Dense(32) → Dense(actions)
```

**Also Good**: Constant size
```python
Dense(64) → Dense(64) → Dense(actions)
```

**Less Optimal**: Increasing size
```python
Dense(32) → Dense(64) → Dense(128)  # Not recommended
```

### 2. Activation Functions

- **Hidden layers**: Use `relu` (fast, effective)
- **Output layer**: Use `linear` (for Q-values)
- **Initialization**: Use `he_uniform` for ReLU

### 3. Loss Functions

- **Huber loss**: Robust to outliers (recommended)
- **MSE**: Simpler but less stable
- **MAE**: More stable but slower convergence

### 4. Dropout

Add dropout for very large models to prevent overfitting:

```python
Dense(256, activation='relu'),
Dropout(0.2),  # 20% dropout
Dense(128, activation='relu'),
```

### 5. Regularization

For complex problems, add L2 regularization:

```python
from tensorflow.keras.regularizers import l2

Dense(128, activation='relu', kernel_regularizer=l2(0.01))
```

## Performance Tips

### Training Speed vs Accuracy

| Architecture Size | Training Speed | Final Performance | Best For |
|------------------|----------------|-------------------|----------|
| Small (990 params) | ⚡⚡⚡⚡ Fast | ⭐⭐⭐ Good | Quick tests |
| Medium (2.9K params) | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ Very Good | Balanced |
| Large (11.8K params) | ⚡⚡ Medium | ⭐⭐⭐⭐⭐ Excellent | **Default** |
| Extra Large (44K params) | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | Complex problems |

### Recommendations by Use Case

**Quick Experiments** (< 5 minutes):
- Use: Small (24-24)
- Episodes: 500
- Purpose: Test hyperparameters

**Production Training** (15-30 minutes):
- Use: Large (128-64-32) ⭐ **Recommended**
- Episodes: 1500
- Purpose: Best results

**Research/Competition** (30-60 minutes):
- Use: Extra Large (256-128-64)
- Episodes: 2000
- Purpose: Maximum performance

## Troubleshooting

### Architecture Not Appearing in GUI

1. Check file is in `model_architectures/` directory
2. Ensure class inherits from `ModelArchitecture`
3. Restart the application
4. Check for syntax errors: `python -c "from model_architectures import ModelFactory; ModelFactory.list_architectures()"`

### Model Training Too Slow

- Switch to smaller architecture (Medium or Small)
- Reduce episodes
- Use "Hide Visualization" option

### Model Not Learning Well

- Switch to larger architecture (Large or Extra Large)
- Increase episodes
- Check reward function is working
- Verify state representation

## Testing

Run the architecture comparison test:

```bash
python test_architectures.py
```

This will:
- List all available architectures
- Show parameter counts
- Display complexity ratings
- Verify all models load correctly

## Summary

The modular architecture system provides:

✅ **Flexibility**: Easy to add new architectures  
✅ **Comparison**: Track which architecture performs best  
✅ **Experimentation**: Test different designs quickly  
✅ **Maintainability**: Each architecture in separate file  
✅ **Automation**: Auto-discovery, no registration code  

🚀 **Ready to use!** Start with **Large (128-64-32)** for best results.
