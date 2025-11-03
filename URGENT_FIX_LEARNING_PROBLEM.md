# URGENT FIX: Model Not Learning (225 steps vs 7 optimal)

## Problem

Fresh training for 3 discs resulted in:
- **Best average**: 225 steps
- **Optimal**: 7 steps
- **Ratio**: 32x worse than optimal! ❌

This is catastrophically bad - model is essentially random.

---

## Root Causes Identified

### 1. **Over-Aggressive Penalty Scaling**
```python
OLD: penalty_scale = 2.0, oscillation_penalty_scale = 3.0
```
- Penalties got scaled 2-3×, drowning out positive rewards
- Success reward (+100) became meaningless compared to scaled penalties

### 2. **Too Much Credit Assignment**
```python
OLD: Propagate to 5 steps back, 50% factor, apply 50% of propagated penalty
```
- Created massive noise in training data
- Agent couldn't distinguish actual cause from propagated effects
- Memory filled with artificially adjusted experiences

### 3. **Prioritized Replay Too Biased**
```python
OLD: 50% penalties, 30% neutral, 20% rewards
```
- Agent spent most time learning what NOT to do
- Forgot successful strategies
- Couldn't learn optimal path

### 4. **Insufficient Exploration**
```python
OLD: epsilon = 0.5 (start), epsilon_decay = 0.99 (fast)
```
- Started with only 50% exploration
- Decayed too quickly
- Agent never discovered good solutions

### 5. **Too Aggressive Oscillation Breaking**
```python
OLD: Threshold = 6 steps, force 10 random actions, boost epsilon +0.3
```
- Interrupted learning too early
- Forced too much randomness
- Destroyed learning momentum

---

## Changes Made

### 1. ✅ Reduced Penalty Scaling
```python
NEW: 
penalty_scale = 1.2              # Was 2.0 - minimal scaling
oscillation_penalty_scale = 1.5  # Was 3.0 - less aggressive
```

**Why**: Let environment rewards speak for themselves. The reward system is already well-designed.

### 2. ✅ Much More Conservative Credit Assignment
```python
NEW:
- Only propagate for SEVERE penalties (≤ -30, was -10)
- Only 2 steps back (was 5)
- Only 20% of propagated penalty (was 50%)
- Propagation factor 0.2 (was 0.5)
```

**Why**: Reduce noise, let agent learn from direct experience.

### 3. ✅ Balanced Experience Replay
```python
NEW: 30% penalties, 30% neutral, 40% rewards
OLD: 50% penalties, 30% neutral, 20% rewards
```

**Why**: Agent needs to learn successful strategies MORE than failures.

### 4. ✅ Proper Exploration
```python
NEW:
epsilon = 1.0              # Was 0.5 - start with full exploration
epsilon_decay = 0.995      # Was 0.99 - slower decay
learning_rate = 0.001      # Was 0.003 - more stable
batch_size = 32            # Was 64 - faster updates
gamma = 0.95               # Was 0.90 - better long-term planning
```

**Why**: Agent needs to explore state space to find optimal solutions.

### 5. ✅ Less Aggressive Oscillation Detection
```python
NEW:
oscillation_threshold = 8  # Was 6 - more patient
force_random_actions = 5   # Was 10 - less disruption
epsilon_boost = 0.1        # Was 0.3 - smaller boost
```

**Why**: Don't interrupt learning prematurely.

---

## Expected Results After Fix

### Training Metrics:

| Episode Range | Success Rate | Avg Steps | Notes |
|--------------|--------------|-----------|-------|
| 1-100 | 0-30% | 50-200 | Exploration phase |
| 100-300 | 30-60% | 20-50 | Learning phase |
| 300-500 | 60-85% | 10-20 | Optimization phase |
| 500-1000 | 85-95%+ | 7-12 | Converged |

**Target**: 90%+ success rate, 8-10 avg steps by episode 1000

### What "Good Training" Looks Like:

```
Episode 100: Steps=45, Success=Yes, Reward=55.7
Episode 200: Steps=28, Success=Yes, Reward=75.3
Episode 300: Steps=15, Success=Yes, Reward=88.2
Episode 400: Steps=11, Success=Yes, Reward=92.1
Episode 500: Steps=9, Success=Yes, Reward=95.4
Episode 700: Steps=7, Success=Yes, Reward=98.9  ← OPTIMAL!
Episode 1000: Steps=8, Success=Yes, Reward=97.2
```

---

## How to Test the Fix

### 1. **Clean Start**
```bash
# Delete old poorly trained model
rm -rf models/dqn_model_*

# Start fresh training
./start_gui.sh
```

### 2. **Training Configuration**
```
Model Architecture: Medium (64-32) or Large (128-64-32)
Number of Discs: 3
Training Episodes: 1000
Batch Size: 32 (default)
```

### 3. **Monitor Progress**

Watch for these **good signs** ✅:
```
Episode 50: Success=Yes, Steps=35
Episode 100: Success=Yes, Steps=22
Episode 200: Success=Yes, Steps=13
Episode 300: Success=Yes, Steps=9
Episode 500: Success=Yes, Steps=8
```

Watch for these **bad signs** ❌:
```
Episode 100: Success=No, Steps=150
Episode 200: Success=No, Steps=200
Episode 300: Success=Yes, Steps=75  ← Still too high
```

If you still see bad signs, the reward system itself might need simplification.

---

## What We Removed vs Kept

### ❌ Removed (Too Aggressive):
- Heavy penalty scaling (2-3×)
- Deep credit assignment (5 steps)
- Penalty-focused replay (50%)
- Aggressive oscillation forcing

### ✅ Kept (Still Useful):
- Episode trajectory tracking
- Credit assignment (but very conservative)
- Balanced experience replay
- Oscillation detection (but gentler)

---

## If Still Not Working

If after these changes you still see >50 avg steps by episode 500, the **environment reward system** is too complex. Consider:

### Option A: Simplify Reward System

```python
# Simple reward system (in toh.py):
if invalid_move:
    reward = -10
elif done:
    reward = 200 - steps  # Reward efficiency
else:
    reward = -0.1  # Small step penalty
    if len(state[2]) > prev_discs_on_goal:
        reward += 10  # Bonus for progress
```

### Option B: Use Traditional DQN Settings

```python
# Disable all enhancements in dqn_agent.py:
penalty_scale = 1.0                    # No scaling
trajectory_penalty_propagation = 0.0   # No credit assignment
# Use random sampling in replay() instead of stratified
```

---

## Key Takeaway

**"Less is More"** in reinforcement learning!

The original credit assignment idea was sound, but:
- ❌ Too aggressive implementation overwhelmed the learning
- ❌ Penalties dominated, agent forgot successes
- ❌ Noise from propagation confused the agent

The fix:
- ✅ Minimal intervention - let environment rewards work
- ✅ Balanced learning from both success and failure
- ✅ Proper exploration before exploitation
- ✅ Stable, gradual learning

---

## Quick Reference

### Before (WRONG):
```
Penalties: 2-3× scaled, propagated 5 steps, 50% of batch
Exploration: Start 50%, decay fast
Result: Agent confused, can't learn
```

### After (CORRECT):
```
Penalties: 1.2-1.5× scaled, propagated 2 steps (rare), 30% of batch  
Exploration: Start 100%, decay slow
Result: Agent explores, learns, optimizes
```

---

## Training Checklist

- [ ] Delete old models
- [ ] Use Medium or Large architecture
- [ ] Train for 1000 episodes
- [ ] Watch avg steps decrease over time
- [ ] Target: <15 steps by episode 500
- [ ] Target: <10 steps by episode 1000
- [ ] Success rate: >90% by episode 1000

If you hit these targets, the fix worked! 🎯
