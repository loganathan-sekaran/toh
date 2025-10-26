# Model Evaluation & Learning Rate Tracking System

## Overview

This document describes the comprehensive model evaluation and learning rate tracking system added to the Tower of Hanoi RL Trainer. This system allows you to:

1. **Measure model efficiency** - Is your model learning effectively?
2. **Compare different models** - Which training configuration works best?
3. **Track learning progress** - How fast does the model converge?
4. **Test models anytime** - Evaluate saved models without retraining

## Key Features

### 1. Automatic Evaluation During Training

When you train a model (Option 2), the system now:

- **Periodic Quick Evaluations** (every 50 episodes)
  - Runs 50 test episodes without exploration
  - Reports success rate, efficiency score, and optimal solves
  - Helps you see learning progress during training

- **Comprehensive Final Evaluation** (after training completes)
  - Runs 100 test episodes for accurate metrics
  - Generates detailed performance report
  - Tracks learning progress windows (100/500/1000 episodes)
  - Finds convergence point (when model reaches 80% success)
  - Saves session data for future comparison

### 2. Model Comparison (Menu Option 5)

Compare multiple trained models side-by-side:

```
📊 Comparing Models
===========================================

Available Models:
1. toh_3discs_20250121_143022.weights.h5 (2025-01-21 14:30:22)
2. toh_3discs_20250121_150533.weights.h5 (2025-01-21 15:05:33)
3. toh_3discs_20250121_161144.weights.h5 (2025-01-21 16:11:44)

Enter model numbers to compare (space-separated, e.g., '1 2 3'):
> 1 2 3
```

**Comparison Output:**
- Side-by-side metrics table
- Winner determination with weighted scoring
- Highlights best values for each metric

**Scoring System:**
- 40% weight on Success Rate
- 30% weight on Efficiency Score
- 30% weight on Optimal Solves

### 3. Learning Reports (Menu Option 6)

View and compare learning rates across training sessions:

```
📈 Learning Reports
===========================================

Available Training Sessions:
1. Training_20250121_143022 - 1000 episodes
2. Training_20250121_150533 - 1500 episodes
3. Training_20250121_161144 - 2000 episodes
```

**Report Contents:**
- Total training time
- Episodes to convergence (80% success rate)
- Average learning rate (episodes/min)
- Success rate progression
- Efficiency score trends
- Step distribution statistics

### 4. Standalone Model Testing (test_model.py)

Test any saved model from the command line:

```bash
# Test a specific model
python test_model.py --model models/toh_3discs_20250121_143022.weights.h5

# Compare multiple models
python test_model.py --compare models/model1.weights.h5 models/model2.weights.h5

# Test with visualization
python test_model.py --model models/my_model.weights.h5 --visualize

# List available models
python test_model.py --list

# Interactive mode
python test_model.py
```

## Performance Metrics Explained

### Success Rate
**Definition:** Percentage of episodes where the model successfully solved the puzzle.

**Interpretation:**
- 90-100%: Excellent - Model has mastered the task
- 70-89%: Good - Model is learning well
- 50-69%: Fair - Model needs more training
- <50%: Poor - May need hyperparameter tuning

### Average Steps
**Definition:** Mean number of moves taken to complete the puzzle.

**For 3 discs:**
- 7 steps: Optimal solution (2^3 - 1 = 7)
- 7-10 steps: Very good
- 10-15 steps: Good
- 15-25 steps: Fair
- >25 steps: Inefficient

### Efficiency Score
**Definition:** (Optimal Steps / Actual Steps) × 100%

**Formula:** efficiency = (7 / avg_steps) × 100

**Interpretation:**
- 100%: Perfect - Always uses optimal moves
- 80-99%: Excellent - Very close to optimal
- 60-79%: Good - Reasonably efficient
- 40-59%: Fair - Room for improvement
- <40%: Poor - Very inefficient

### Optimal Solves
**Definition:** Number of episodes solved in exactly 7 steps (for 3 discs).

**Interpretation:**
- High optimal solves = Model has learned the perfect strategy
- Low optimal solves = Model succeeds but doesn't know the optimal path

### Near-Optimal Solves
**Definition:** Episodes solved within 20% of optimal (≤ 8-9 steps for 3 discs).

**Interpretation:**
- Shows how often the model gets "close enough"
- Good metric for practical applications

### Convergence Episode
**Definition:** Episode number where success rate first reaches 80%.

**Interpretation:**
- Lower is better - faster learning
- Use to compare different training configurations
- Track impact of hyperparameter changes

### Step Distribution
**Statistics:** Min, Median, Max, Standard Deviation

**Interpretation:**
- Low std dev = Consistent performance
- High std dev = Unpredictable behavior
- Median closer to min = More efficient on average

## Tracking Learning Progress

The system analyzes learning in three windows:

1. **First 100 Episodes** - Early learning phase
   - Shows how quickly the model picks up basic patterns
   
2. **First 500 Episodes** - Mid-training learning
   - Indicates if the model is converging
   
3. **First 1000 Episodes** - Long-term learning
   - Shows final performance trajectory

**Learning Rate Calculation:**
```
learning_rate = episodes_processed / training_time_minutes
```

Higher learning rate with same success = more efficient training.

## Files Generated

### During Training
1. **Model Weights:** `models/toh_3discs_YYYYMMDD_HHMMSS.weights.h5`
   - Saved every checkpoint_interval episodes
   - Final model saved at end

2. **Evaluation Report:** `models/evaluation_report_YYYYMMDD_HHMMSS.json`
   - Comprehensive metrics in JSON format
   - Can be loaded and compared later

3. **Training Session:** `models/learning_reports/session_YYYYMMDD_HHMMSS.json`
   - Complete training session data
   - Episode-by-episode history
   - Convergence analysis

### Directory Structure
```
models/
├── toh_3discs_20250121_143022.weights.h5
├── toh_3discs_20250121_150533.weights.h5
├── evaluation_report_20250121_143022.json
├── evaluation_report_20250121_150533.json
└── learning_reports/
    ├── session_20250121_143022.json
    └── session_20250121_150533.json
```

## UI Updates in Fast Mode

**Previous Issue:** When visualization was hidden (fast mode), the UI appeared frozen.

**Solution:** The system now **always updates** episode count, epsilon, and success rate, even in fast mode.

**What You'll See:**
- Episode counter increments smoothly
- Success rate updates in real-time
- Epsilon decay visible
- Periodic evaluation results printed

**Performance:** UI updates are lightweight and don't slow training.

## Best Practices

### Training
1. **Start with default settings** (1000 episodes)
2. **Monitor success rate** - Should reach >70% within 500 episodes
3. **Use fast mode** for bulk training, show visualization when debugging
4. **Save multiple models** with different hyperparameters

### Evaluation
1. **Run final evaluation** to get accurate metrics
2. **Compare at least 3 models** before choosing "best"
3. **Look at multiple metrics** - don't just optimize for success rate
4. **Check convergence episode** to optimize training time

### Model Selection
1. **High success rate** (>90%) is priority
2. **High efficiency score** (>80%) means optimal strategy
3. **Low convergence episode** (<300) means fast learning
4. **Consistent step distribution** (low std dev) is desirable

## Troubleshooting

### "No models found"
- Train at least one model first (Option 2)
- Check that `models/` directory exists
- Verify .weights.h5 files are present

### "Need at least 2 models to compare"
- Train multiple models with different settings
- Each training session saves a new model file

### Evaluation takes too long
- Reduce `num_episodes` parameter (default: 100)
- 50 episodes is usually sufficient for comparison
- 100+ episodes for publication-quality metrics

### UI still not updating in fast mode
- Make sure you have the latest visualizer.py
- Check that PyQt6 is properly installed
- Try toggling visualization ON then OFF again

## Example Workflow

### 1. Initial Training
```
Menu > 2 (Train)
Episodes: 1000
Watch training or toggle to fast mode
```

### 2. Quick Iteration
```
Menu > 4 (Quick Train)
Runs 500 episodes in fast mode
Repeat 2-3 times with different random seeds
```

### 3. Compare Results
```
Menu > 5 (Compare Models)
Select: 1 2 3
Review comparison table
Identify best model
```

### 4. Analyze Learning
```
Menu > 6 (Learning Reports)
Compare training sessions
Identify fastest learning configuration
```

### 5. Test Best Model
```
Menu > 3 (Test)
Select best model from comparison
Watch it solve optimally with animation
```

## Future Improvements

Possible enhancements for the evaluation system:

1. **Automated hyperparameter search** - Try multiple configurations automatically
2. **Visualization of learning curves** - Plot metrics over time
3. **Cross-validation** - Test on different disc counts
4. **Transfer learning** - Train on 3 discs, test on 4+ discs
5. **Ensemble methods** - Combine multiple models
6. **Real-time comparison dashboard** - Live metric comparison during training

## Technical Details

### ModelEvaluator Class
```python
evaluator = ModelEvaluator(env, num_discs=3, optimal_steps=7)
report = evaluator.evaluate_model(agent, num_episodes=100)
```

**Methods:**
- `evaluate_model()` - Run greedy evaluation
- `generate_report()` - Create human-readable report
- `compare_models()` - Side-by-side comparison
- `track_learning_progress()` - Analyze convergence
- `save_report()` / `load_report()` - Persistence

### LearningRateTracker Class
```python
tracker = LearningRateTracker()
tracker.save_training_session(session_data, report, history)
tracker.compare_learning_rates()
```

**Methods:**
- `save_training_session()` - Store session data
- `load_session()` - Retrieve session data
- `list_sessions()` - Show all sessions
- `compare_learning_rates()` - Multi-session comparison

### Metrics Calculation

**Success Rate:**
```python
success_rate = (successful_episodes / total_episodes) * 100
```

**Efficiency Score:**
```python
efficiency_score = (optimal_steps / avg_steps) * 100
```

**Convergence Episode:**
```python
# First episode where rolling 100-episode success rate >= 80%
for i in range(100, len(history)):
    window_success = sum(history[i-100:i]) / 100
    if window_success >= 0.8:
        convergence_episode = i
        break
```

## Conclusion

The evaluation system provides comprehensive tools to:
- ✅ Measure if your model is learning efficiently
- ✅ Compare different training configurations
- ✅ Track learning progress over time
- ✅ Test models anytime with or without visualization
- ✅ Identify optimal training settings

Use these tools iteratively to improve your model's learning rate and final performance!
