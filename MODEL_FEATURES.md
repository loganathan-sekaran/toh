## Model Visualization and Management Features

### Overview
New features for visualizing DQN model architecture, managing trained models, and tracking learning performance.

### New Components

#### 1. **Model Architecture Visualizer** (`model_visualizer.py`)
- **Visual Display**: Shows neural network layers, nodes, and connections
- **Auto-Update**: Refreshes when model changes
- **Layer Information**: Displays layer types, node counts, and total parameters
- **Color Coding**:
  - Green: Input layer
  - Blue: Hidden layers
  - Orange: Output layer

#### 2. **Model Manager** (`model_manager.py`)
- **Auto-Save**: Models automatically saved after training
- **Metadata Tracking**: Stores training metrics with each model:
  - Training episodes
  - Success rate
  - Average steps to solution
  - Epsilon value
  - Creation timestamp
- **Model Directory**: All models saved to `models/` directory
- **Format**: Uses TensorFlow Keras `.keras` format

#### 3. **Model Selection Dialog** (`model_selection_dialog.py`)
- **Browse Models**: Table view of all trained models
- **Performance Metrics**: Shows success rate, avg steps, epsilon for each model
- **Auto-Select Latest**: Defaults to most recently trained model
- **Delete Models**: Remove unwanted models
- **Detailed View**: Shows full metadata for selected model

### Updated GUI Features

#### Training
- **Auto-Save**: Models automatically saved after training completes
- **Performance Tracking**: Success rate and average steps tracked during training
- **Architecture Display**: Option to view model architecture after training

#### Testing
- **Model Selection**: Browse and select any trained model
- **Architecture Viewer**: See model structure before testing
- **Visual Test**: Watch model solve puzzle with visualization
- **Performance Report**: Shows solution steps and success

### Usage

#### Train a Model
```python
# Via GUI
1. Click "Train Model"
2. Configure episodes (e.g., 100)
3. Watch training with visualization
4. Model automatically saved at completion
5. Optional: View architecture

# Model saved to: models/dqn_model_YYYYMMDD_HHMMSS/
```

#### Test a Model
```python
# Via GUI
1. Click "Test Model"
2. Select model from list (latest selected by default)
3. View model architecture
4. Watch model solve puzzle
5. See performance metrics
```

#### View Model Architecture
```python
from model_visualizer import ModelVisualizerWidget
from model_manager import ModelManager

# Load a model
manager = ModelManager()
agent, metadata = manager.load_model("dqn_model_20251026_143000")

# Create visualizer
viz = ModelVisualizerWidget()
viz.set_model(agent.model, f"Success: {metadata['success_rate']:.1f}%")
viz.show()
```

### Model Metadata Structure
```json
{
  "name": "dqn_model_20251026_143000",
  "created_at": "2025-10-26T14:30:00",
  "state_size": 27,
  "action_size": 9,
  "epsilon": 0.123,
  "gamma": 0.95,
  "learning_rate": 0.001,
  "episodes": 100,
  "num_discs": 3,
  "success_rate": 87.5,
  "avg_steps": 24.3,
  "total_steps": 2430
}
```

### Performance Tracking

Models are evaluated on:
1. **Learning Rate**: How quickly success rate improves
2. **Convergence Speed**: Episodes needed to reach consistent success
3. **Solution Efficiency**: Average steps to solve puzzle
4. **Final Performance**: Success rate and epsilon at completion

### Model Comparison

Compare models by:
- **Success Rate**: Higher is better
- **Avg Steps**: Lower is better (optimal is 7 for 3 discs)
- **Epsilon**: Lower indicates more learning (less exploration)
- **Training Time**: Episodes needed to train

### File Structure
```
toh/
├── models/                          # Saved models directory
│   ├── dqn_model_20251026_143000/
│   │   ├── model.keras              # Keras model
│   │   └── metadata.json            # Training metadata
│   └── dqn_model_20251026_150000/
│       ├── model.keras
│       └── metadata.json
├── model_visualizer.py              # Architecture visualization widget
├── model_manager.py                 # Model persistence and loading
├── model_selection_dialog.py        # Model browser dialog
└── gui_launcher.py                  # Updated with model features
```

### Benefits

1. **Track Progress**: See which models perform best
2. **Understand Architecture**: Visualize network structure
3. **Easy Testing**: Load and test any model instantly
4. **Performance Insights**: Compare learning rates and efficiency
5. **Model Management**: Organize and delete old models

### Future Enhancements

Potential additions:
- Model versioning
- Training history graphs
- Hyperparameter comparison
- Export model statistics
- Batch testing across multiple models
