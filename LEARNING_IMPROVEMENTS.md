# Learning Improvements for Tower of Hanoi RL

## Problem
After 141+ episodes with the large model, the agent was not learning and still taking 1000 steps to solve the puzzle (or failing). Agent was repeating moves that gave negative rewards.

## Root Causes Identified

### 1. **Insufficient Training Frequency** (CRITICAL)
- **Before**: Agent trained only ONCE per episode
- **After**: Agent trains after EVERY step
- **Impact**: Dramatically increases learning opportunities from ~500 training calls to ~500,000 training calls over 500 episodes

### 2. **No Action Masking** (CRITICAL - NEW FIX)
- **Problem**: Agent wasted learning on invalid moves, even when it "knew" they were bad
- **Solution**: Added action masking to only allow valid moves
- **How it works**:
  - `env.get_valid_actions()` returns list of valid action indices
  - `agent.act(state, valid_actions)` only chooses from valid moves
  - During exploration: randomly picks from valid actions only
  - During exploitation: masks Q-values of invalid actions with -inf
- **Impact**: Agent never wastes time learning from invalid moves, focuses only on finding optimal paths
- **Invalid move penalty**: Increased from -10 to -50 (in case action masking is bypassed)

### 3. **Sparse Rewards**
- **Before**: Only rewarded at completion (+100) or invalid moves (-10), with -1 per step
- **After**: Added reward shaping:
  - +5 bonus for moving a disc to the goal rod
  - +2 additional bonus for correct placement on goal rod
  - Helps agent learn the sub-goals of the Tower of Hanoi puzzle

### 4. **Suboptimal Hyperparameters**
- **Learning Rate**: Increased from 0.0005 → 0.001 (faster learning)
- **Batch Size**: Reduced from 64 → 32 (more frequent weight updates)
- **Epsilon Decay**: Slowed from 0.995 → 0.998 (more exploration)
- **Gamma**: Reduced from 0.99 → 0.95 (focus more on immediate rewards)
- **Target Network Update**: Slowed from every 10 → every 100 steps (more stability)

## Expected Results

With these changes, you should see:

1. **Faster Learning**: Steps should decrease noticeably within first 50-100 episodes
2. **Better Success Rate**: More episodes should reach completion (all discs on rod 2)
3. **Epsilon Decay**: Watch epsilon value drop as agent learns (shown in UI)
4. **Reward Trends**: Total reward per episode should increase over time

## Monitoring Training Progress

Key metrics to watch in the UI:
- **Steps per episode**: Should trend downward
- **Success Rate**: Should trend upward
- **Epsilon**: Should gradually decrease from 1.0 → 0.01
- **Reward**: Should increase as agent gets better

For 3 discs, optimal solution is 7 steps (2^3 - 1). Agent should learn to solve in 7-15 steps consistently after ~200-300 episodes.

## Training Tips

1. **Start with visualization OFF** for first 100-200 episodes (faster training)
2. **Turn visualization ON** after ~200 episodes to watch learned behavior
3. **Save model** when success rate > 80% and avg steps < 20
4. **Increase to 4 discs** only after mastering 3 discs (optimal = 15 steps)

## Technical Details

### Training Loop Changes
```python
# OLD: Train once per episode, no action masking
for episode in episodes:
    while not done:
        action = agent.act(state)  # Can choose invalid actions!
        # ... game loop ...
    agent.replay()  # Only once!

# NEW: Train after every step WITH action masking
for episode in episodes:
    while not done:
        valid_actions = env.get_valid_actions()  # Get valid moves
        action = agent.act(state, valid_actions)  # Only choose valid!
        # ... execute action ...
        agent.remember(...)
        if len(memory) > batch_size:
            agent.replay()  # Every step!
```

### Action Masking Implementation
```python
# In agent.act():
if valid_actions provided:
    if exploring:
        return random.choice(valid_actions)  # Only from valid
    else:
        masked_q = [-inf, -inf, -inf, ...]  # Mask invalid
        masked_q[valid_actions] = q_values[valid_actions]
        return argmax(masked_q)  # Best valid action
```

### Reward Shaping
```python
# Added intermediate rewards to guide learning:
if moved_to_goal_rod:
    reward += 5  # Encourage moving to goal
if correct_position_on_goal:
    reward += 2  # Encourage correct stacking
```

This creates a dense reward signal that helps the agent learn the hierarchical nature of Tower of Hanoi.
