# Reward System Fix - November 3, 2024

## Problem Identified

Training was still failing with **222 average steps** (should be 7-10). The issue was **reward magnitudes were too large**, making learning unstable.

### Root Causes

1. **Step penalty accumulation**: `-0.1` per step × 222 steps = -22.2 penalty
   - Made rewards vary wildly: +68.9, +89.9, -47.1, etc.
   - Agent couldn't learn consistent Q-values

2. **Completion rewards too high**: Up to +200 for completion
   - Reward clipping allowed up to +300
   - Created massive value spikes

3. **Individual rewards too large**:
   - Correct placement: +30 (now +10)
   - Largest disc to target: +20-25 (now +8)
   - Removing correct disc: -40 (now -15)
   - Moving largest to wrong rod: -15 (now -5)

4. **Flawed efficiency calculation**: Used `self.max_steps` instead of optimal formula
   - Should be `2^n - 1` for n discs

## Changes Applied to `toh.py`

### 1. Removed Step Penalty (Line 60)
```python
# BEFORE
reward = -0.1  # Small base penalty to encourage efficiency

# AFTER  
reward = 0  # No base penalty - rely on completion bonus for efficiency
```

**Why**: Step penalties accumulate unpredictably over long episodes, making rewards inconsistent.

### 2. Scaled Down All Rewards (Various Lines)

| Reward Type | Before | After | Reduction |
|------------|--------|-------|-----------|
| Correct placement | +30 | +10 | 66% |
| Remove correct disc | -40 | -15 | 62% |
| Largest to target | +20-25 | +8 | 68% |
| Largest to wrong rod | -15 | -5 | 67% |
| Move to goal bonus | +5 | +2 | 60% |
| Correct position bonus | +3 | +1 | 67% |
| Maintaining correct state | +2 | +1 | 50% |

### 3. Fixed Completion Rewards (Lines 224-242)

```python
# BEFORE
optimal_steps = self.max_steps  # WRONG - this is 100!
# ... complex logic with +200, +100, +80, +50

# AFTER
optimal_steps = (2 ** self.num_discs) - 1  # Correct: 7 for 3 discs
if self.steps <= optimal_steps:
    reward += 50  # Perfect solution
elif self.steps <= optimal_steps * 1.5:
    reward += 30  # Good solution
elif self.steps <= optimal_steps * 2:
    reward += 20  # Acceptable
else:
    reward += 10  # Completed but inefficient
```

**Why**: 
- Uses correct optimal formula (7 steps for 3 discs, not 100)
- Completion bonus now +50 max (was +200)
- Simple tiered system based on efficiency

### 4. Reduced Reward Clipping (Line 244)

```python
# BEFORE
reward = np.clip(reward, -100, 300)

# AFTER
reward = np.clip(reward, -100, 100)
```

**Why**: Prevents extreme positive values that destabilize learning.

## Expected Impact

### Reward Ranges (Per Step)
- **Before**: -70 to +200+ (huge variance)
- **After**: -50 to +20 typical, +50 max for completion

### Training Behavior
- **More stable Q-values**: Rewards are consistent and predictable
- **Faster convergence**: Agent can distinguish good from bad actions
- **Better generalization**: Simpler reward structure = clearer learning signal

### Success Criteria
After these changes, expect:
- Episode 100: 50-70% success, 20-30 avg steps
- Episode 500: 85-95% success, 8-12 avg steps  
- Episode 1000: 95%+ success, 7-9 avg steps

## Testing Instructions

1. **Delete old models**: `rm -rf models/dqn_model_*`
2. **Start fresh training**:
   ```bash
   ./start_gui.sh
   # Select "Train Model"
   # Architecture: Large (128-64-32)
   # Discs: 3
   # Episodes: 1000
   ```

3. **Monitor rewards**: Should see consistent small values like:
   ```
   Episode 50, Step 5: REWARD = 10.0 (correct placement)
   Episode 50, Step 12: PENALTY = -15.0 (removed correct disc)
   Episode 50, Step 15: REWARD = 50.0 (COMPLETED!)
   ```

4. **Check progress**: By episode 500, avg steps should be ~10

## Why This Fixes the Problem

The original issue was **signal-to-noise ratio**:
- With `-0.1` per step, a 100-step episode had -10 base penalty
- Add +30 for placements, -40 for mistakes, +200 for completion
- Result: Total rewards ranged from -100 to +300+ randomly

This is like trying to hear a whisper (optimal strategy signal) in a stadium (massive reward swings).

**New system**:
- No accumulating step penalty = consistent baseline
- Smaller individual rewards (±5 to ±15) = clearer distinctions
- Reasonable completion bonus (+50) = proper goal incentive
- Result: Total rewards -50 to +100 with clear patterns

The agent can now **actually learn** because:
1. Good actions consistently give +8 to +10
2. Bad actions consistently give -12 to -15  
3. Completion gives +50 as a clear goal signal
4. No random accumulation masks the learning signal

## Rollback Plan

If training still doesn't work after 500 episodes:

**Option A**: Disable credit assignment completely
```python
# In dqn_agent.py, line 38
self.trajectory_penalty_propagation = 0.0  # Disable completely
```

**Option B**: Simplify rewards even more
```python
# In toh.py, make rewards binary
# Correct placement: +5
# Wrong move: -5  
# Completion: +20
```

**Option C**: Use sparse rewards only
```python
# Only reward at completion, nothing else
# This forces agent to learn purely from trial and error
```
