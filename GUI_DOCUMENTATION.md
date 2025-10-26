# GUI Launcher Documentation

## Overview

The Tower of Hanoi RL Trainer now features a modern **PyQt6 graphical user interface** that replaces the command-line menu. All functionality is accessible through intuitive buttons and dialogs.

## Launching the GUI

### Default Launch
Simply run the main script without arguments:
```bash
./start.sh
# or
python main.py
```

The GUI will launch automatically.

### Alternative Methods
```bash
# Direct GUI launch
./start_gui.sh
# or
python gui_launcher.py

# Force CLI menu (old behavior)
python main.py --cli
# or
python main.py cli
```

## GUI Features

### Main Menu Window

The main launcher window provides 8 buttons for all major functions:

#### 🎬 Demo
- **Function**: Watch the optimal solution with animation
- **Use Case**: See how the puzzle should be solved perfectly
- **Action**: Opens visualization window showing 7-move optimal solution

#### 🏋️ Train Model
- **Function**: Train a new AI model with visualization
- **Dialog**: Opens configuration dialog to set:
  - Number of discs (3-5)
  - Training episodes (100-10,000)
  - Batch size (16-128)
- **Default**: 3 discs, 1000 episodes, batch size 32
- **Features**: 
  - Progress visualization during training
  - Toggle button to hide/show animation
  - Real-time metrics updates
  - Automatic model saving

#### 🧪 Test Model
- **Function**: Test a trained model with visualization
- **Dialog**: Opens model selection dialog
  - Lists all available models with timestamps
  - Shows creation date and time
- **Action**: Runs 50 test episodes and visualizes agent playing
- **Output**: 
  - Success rate and average steps
  - Efficiency score
  - Live visualization of agent solving puzzle

#### ⚡ Quick Train
- **Function**: Fast training without visualization
- **Settings**: 500 episodes, 3 discs, batch size 32
- **Use Case**: Quickly train multiple models for comparison
- **Confirmation**: Shows dialog before starting
- **Speed**: Much faster than regular training (no animation overhead)

#### 📊 Compare Models
- **Function**: Compare performance of multiple models side-by-side
- **Dialog**: Model comparison interface
  - Multi-select list (Ctrl+Click to select multiple)
  - Minimum 2 models required
  - Compare button runs evaluation
- **Output**: Comparison table showing:
  - Success rate
  - Average steps
  - Efficiency score
  - Optimal solves
  - Step distribution statistics
  - Winner determination with weighted scoring

#### 📈 Learning Reports
- **Function**: View training session history and learning rates
- **Display**: Shows all past training sessions with:
  - Session name and timestamp
  - Total episodes trained
  - Training duration
  - Convergence episode (when 80% success achieved)
  - Learning rate (episodes per minute)
  - Final performance metrics
- **Use Case**: Track improvement over time, compare training configurations

#### 📚 Tutorial
- **Function**: Learn how Tower of Hanoi works
- **Content**:
  - Game rules and objectives
  - Optimal solution explanation
  - How DQN learning works
  - Training process overview
  - Tips for best results
- **Format**: Rich HTML-formatted tutorial with sections

#### ❌ Exit
- **Function**: Close the application
- **Action**: Immediately exits the GUI

## Dialog Windows

### Training Configuration Dialog
**Purpose**: Set training parameters before starting

**Fields**:
- **Number of Discs**: Spinner (3-5)
  - 3 discs: Standard, 7 moves optimal (recommended for beginners)
  - 4 discs: Advanced, 15 moves optimal
  - 5 discs: Expert, 31 moves optimal
  
- **Training Episodes**: Spinner (100-10,000)
  - 100-500: Quick test
  - 1000: Standard training (recommended)
  - 2000+: Extended training for higher accuracy
  
- **Batch Size**: Spinner (16-128)
  - 16: Slower but more stable learning
  - 32: Balanced (recommended)
  - 64-128: Faster training, may be less stable

**Tip Box**: Shows helpful advice about episode selection

**Buttons**: OK / Cancel

### Model Selection Dialog
**Purpose**: Choose a trained model to test

**Display**:
- List of all `.weights.h5` files in `models/` directory
- Each entry shows:
  - Model filename
  - Creation timestamp (extracted from filename)
- Most recent models appear first

**Selection**: Single click to select

**Empty State**: "No models found. Train a model first!"

**Buttons**: OK / Cancel

### Model Comparison Dialog
**Purpose**: Compare multiple models side-by-side

**Interface**:
- **Instructions**: "Select 2 or more models to compare (Ctrl+Click for multiple)"
- **Model List**: Multi-selection enabled
  - Ctrl+Click: Add to selection
  - Ctrl+Click again: Remove from selection
- **Compare Button**: Runs evaluation on all selected models
- **Results Area**: Text display showing comparison table

**Process**:
1. Select models (minimum 2)
2. Click "Compare Selected Models"
3. Button shows "Comparing... Please wait" during evaluation
4. Results appear in text area below
5. Dialog shows when complete

**Output Format**:
```
Model Comparison Report
=======================

Model 1: toh_3discs_20250121_143022
  Success Rate:    95.0%
  Avg Steps:       8.2
  Efficiency:      85.4%
  ...

Model 2: toh_3discs_20250121_150533
  Success Rate:    88.0%
  Avg Steps:       9.5
  Efficiency:      73.7%
  ...

WINNER: Model 1
```

**Error Handling**:
- Warning if less than 2 models selected
- Error dialog if evaluation fails

**Buttons**: Close

### Learning Reports Dialog
**Purpose**: View historical training data

**Display**: Text area showing:
- List of all training sessions
- Session details:
  - Timestamp
  - Episodes trained
  - Training time
  - Convergence metrics
  - Final success rate
  - Comparison across sessions

**Empty State**: "No training reports found. Train a model first to generate reports!"

**Auto-Load**: Reports load automatically when dialog opens

**Buttons**: Close

### Tutorial Dialog
**Purpose**: Educational content about Tower of Hanoi

**Sections**:
1. **The Rules**: Game mechanics and objectives
2. **Optimal Solution**: Math behind minimum moves
3. **How the AI Learns**: DQN explanation
4. **Training Process**: What happens during training
5. **Tips for Best Results**: Practical advice
6. **Try It Yourself**: Call to action

**Format**: Rich HTML with:
- Headings and subheadings
- Bullet points and numbered lists
- Bold emphasis
- Emoji icons
- Mathematical notation (2^N - 1)

**Scrollable**: Content area scrolls if needed

**Buttons**: Close

## GUI Design

### Visual Style
- **Theme**: Fusion style (modern, cross-platform)
- **Colors**:
  - Primary buttons: Blue (#007bff)
  - Hover: Darker blue (#0056b3)
  - Exit button: Red (#dc3545)
  - Background: Light gray (#f5f5f5)
  - Text: Dark gray (#666) for descriptions
  
### Layout
- **Main Window**: 600x500 minimum
- **Padding**: 40px margins for spacious feel
- **Spacing**: 20px between major sections, 15px between buttons
- **Button Height**: 50px minimum for easy clicking

### Typography
- **Title**: 24pt, bold
- **Subtitle**: 12pt
- **Buttons**: 16pt, bold, left-aligned
- **Descriptions**: 11pt, gray, left-aligned with padding

### User Experience
- **Descriptive Labels**: Each button has subtitle explaining function
- **Clear Hierarchy**: Title → Buttons → Exit clearly separated
- **Visual Feedback**: 
  - Hover effects on buttons
  - Press effects for tactile feel
  - Disabled states when appropriate
  - Progress indicators during long operations
  
- **Error Handling**:
  - Friendly warning dialogs
  - Clear error messages
  - Graceful fallbacks

## Integration with Existing Code

### How It Works

The GUI launcher integrates seamlessly with existing training and testing code:

1. **Demo**: Calls `demo_visualizer()` from `main.py`
2. **Train**: Calls `train_with_visualization()` from `train_with_gui.py`
3. **Test**: Calls `load_and_test_model()` from `test_model.py`
4. **Compare**: Calls `compare_models()` from `test_model.py`
5. **Reports**: Uses `LearningRateTracker` from `model_evaluation.py`

### Window Management

**Pattern**: Hide → Execute → Show
```python
self.hide()  # Hide main menu
try:
    # Execute function (may open new window)
    train_with_visualization(...)
finally:
    self.show()  # Restore main menu when done
```

**Benefits**:
- Clean single-window experience
- Main menu returns after each action
- No window stacking issues

### Output Capture

For text-based outputs (comparison, reports), the GUI captures `stdout`:
```python
old_stdout = sys.stdout
sys.stdout = captured_output = io.StringIO()
# ... call function ...
sys.stdout = old_stdout
output = captured_output.getvalue()
# Display in text widget
```

## File Locations

### Source Files
- **Main GUI**: `gui_launcher.py`
- **Launch Scripts**: 
  - `start_gui.sh` (direct GUI)
  - `start.sh` (defaults to GUI)
  - `main.py` (defaults to GUI unless --cli)

### Generated Files
- **Models**: `models/*.weights.h5`
- **Reports**: `models/evaluation_report_*.json`
- **Sessions**: `models/learning_reports/session_*.json`

## Command-Line Options

Even with GUI, command-line usage is still supported:

```bash
# Direct commands (skip GUI)
python main.py demo
python main.py train --episodes 2000
python main.py test --model models/my_model.weights.h5

# Force CLI menu
python main.py --cli
python main.py cli

# Launch GUI explicitly
python gui_launcher.py
./start_gui.sh
```

## Advantages of GUI

### Over CLI Menu
✅ **Visual Clarity**: See all options at once  
✅ **Better Organization**: Grouped by function  
✅ **Richer Feedback**: Dialogs with progress indicators  
✅ **Easier Configuration**: Spinners vs. typing numbers  
✅ **Model Selection**: Visual list vs. remembering filenames  
✅ **Multi-Select**: Easy model comparison selection  
✅ **Formatted Output**: Rich text display for reports  
✅ **Modern UX**: Hover effects, visual hierarchy  
✅ **Less Error-Prone**: Validation in dialogs  
✅ **Tutorial Display**: HTML formatting for better readability  

### Maintained CLI Benefits
✅ **Automation**: Still scriptable via command-line  
✅ **Remote Use**: SSH users can use --cli flag  
✅ **Quick Actions**: Direct commands bypass menu  
✅ **Logging**: stdout still works for scripts  

## Troubleshooting

### GUI Won't Launch
**Problem**: Import errors or blank window

**Solutions**:
1. Check PyQt6 is installed: `pip list | grep PyQt6`
2. Reinstall if needed: `pip install PyQt6`
3. Try CLI mode: `python main.py --cli`
4. Check terminal for error messages

### Models Not Showing
**Problem**: Model selection dialog shows "No models found"

**Solution**:
1. Train at least one model first
2. Check `models/` directory exists
3. Verify `.weights.h5` files are present
4. Check file permissions

### Comparison Fails
**Problem**: Error dialog when comparing models

**Solutions**:
1. Ensure all selected models are valid
2. Check models match disc count (3 discs)
3. Verify models aren't corrupted
4. Try testing each model individually first

### Reports Empty
**Problem**: Learning reports shows "No training reports found"

**Solution**:
1. Complete at least one training session
2. Check `models/learning_reports/` directory exists
3. Verify `.json` files are present
4. Train new model to generate fresh report

## Future Enhancements

Potential GUI improvements:

1. **Real-time Training Graph**: Live plot of success rate over episodes
2. **Model Management**: Rename, delete, organize models
3. **Hyperparameter Presets**: Quick configs (Fast/Balanced/Accurate)
4. **Comparison Plots**: Visual charts for model comparison
5. **Export Reports**: Save comparison results as PDF/CSV
6. **Dark Mode**: Theme toggle for different preferences
7. **Multi-Monitor**: Remember window positions
8. **Drag-and-Drop**: Drag model files to test
9. **Recent Models**: Quick access to recently trained models
10. **Training Queue**: Schedule multiple training runs

## Best Practices

### Workflow Recommendations

1. **First Time Users**:
   - Start with Tutorial button
   - Try Demo to see optimal solution
   - Train with default settings (1000 episodes)
   - Test the trained model

2. **Regular Training**:
   - Use Quick Train for multiple models
   - Compare 3-5 models to find best
   - Review Learning Reports to track progress
   - Adjust parameters based on comparison results

3. **Advanced Users**:
   - Experiment with different disc counts
   - Try various episode counts (500-2000)
   - Compare models with same settings but different random seeds
   - Use reports to optimize learning rate

### Performance Tips

- **Quick Train** is 3-5x faster than regular training (no animation)
- **Hide Visualization** toggle in regular training for speed boost
- Close other applications during training for consistent results
- Use **Compare Models** with 100 episodes for accurate metrics

## Conclusion

The GUI launcher provides a modern, user-friendly interface for all Tower of Hanoi RL Trainer features. It maintains backward compatibility with CLI usage while offering significant UX improvements for interactive use.

**Key Takeaway**: All menu options from the CLI are now accessible through intuitive GUI buttons and dialogs, making the application easier to use without sacrificing functionality.
