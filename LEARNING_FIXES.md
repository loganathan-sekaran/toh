# DQN Learning Fixes - Addressing Oscillation Traps

## Problem Diagnosis

The agent was completely stuck in infinite oscillation loops, repeating the same 2-3 moves hundreds of times despite receiving significant penalties (-10.1 to -20.1 per move). Analysis of training logs revealed:

### Observed Symptoms
- **Episode 2, Steps 431-542**: 112 consecutive oscillations between actions `0→2` and `2→0` with disc 1
- **Episode 2, Steps 697-792**: 96 consecutive oscillations of the same pattern
- Agent ignored penalties ranging from -10.1 to -20.1 per step
- Step counts reaching 1000+ (max limit) instead of optimal 7 steps for 3 discs

### Root Causes Identified

1. **Slow Exploitation**: `epsilon=0.5` with `epsilon_decay=0.995` meant agent did 50% random actions for 500+ episodes, never fully leveraging learned Q-values

2. **Q-Value Saturation**: With penalties in the -10 to -20 range and gamma=0.90, Q-values for oscillating moves converged to similar negative values, making them indistinguishable to the agent

3. **Weak Learning Signal**: `learning_rate=0.001` was too slow to update Q-values fast enough to escape oscillation attractors

4. **No Direct Oscillation Prevention**: While penalties existed in the environment, the agent had no mechanism to detect it was trapped in a loop

5. **Penalties Not Amplified**: Negative rewards were treated the same as positive rewards in memory storage, despite being more critical for avoiding bad states

## Implemented Solutions

### 1. Oscillation Detection and Blocking (CRITICAL)

**File**: `dqn_agent.py`

Added intelligent oscillation detection that monitors recent actions and blocks repeating patterns:

```python
# Added to __init__:
self.recent_actions = deque(maxlen=20)  # Track last 20 actions
self.oscillation_threshold = 8  # Detect if same 2-action pattern repeats 4+ times

# In act() method:
if len(self.recent_actions) >= self.oscillation_threshold:
    last_actions = list(self.recent_actions)[-self.oscillation_threshold:]
    
    # Check for alternating pattern: A,B,A,B,A,B,A,B...
    if len(set(last_actions)) == 2:  # Only 2 unique actions
        is_alternating = all(
            last_actions[i] != last_actions[i+1] 
            for i in range(len(last_actions)-1)
        )
        if is_alternating:
            # Block both oscillating actions
            blocked_actions = set(last_actions)
            print(f"⚠️  OSCILLATION DETECTED: blocking actions {blocked_actions}")
```

**How it works**:
- Tracks last 20 actions taken
- When 8 consecutive actions form a strict alternating pattern (A,B,A,B,A,B,A,B)
- Blocks both actions from Q-value selection by setting their values to `-inf`
- Forces agent to explore different actions, breaking the loop

**Impact**: Prevents agent from being permanently trapped in 2-action loops

### 2. Faster Epsilon Decay

**Changed**: `epsilon_decay = 0.995` → `0.99`

**Effect**:
- Old: Reaches epsilon_min=0.01 in ~460 episodes
- New: Reaches epsilon_min=0.01 in ~90 episodes

**Why**: Agent needs to transition from exploration to exploitation faster. Starting at epsilon=0.5 (already balanced), we want it to reach pure exploitation within 100 episodes so learned Q-values can be fully utilized.

### 3. Increased Learning Rate

**Changed**: `learning_rate = 0.001` → `0.003` (3x increase)

**Why**: 
- Penalties of -10 to -20 need to propagate through Q-network faster
- With slow learning rate, Q-values update too gradually to distinguish bad oscillating moves from less-bad moves
- Faster learning allows neural network to adjust Q-values more aggressively in response to negative rewards

**Trade-off**: May cause some instability initially, but with target network (updated every 100 steps), this is mitigated

### 4. Penalty Scaling in Memory

**Added**: `penalty_scale = 2.0`

```python
def remember(self, state, action, reward, next_state, done):
    # Scale penalties (negative rewards) to amplify learning signal
    if reward < 0:
        scaled_reward = reward * self.penalty_scale
    else:
        scaled_reward = reward
    
    self.memory.append((state, action, scaled_reward, next_state, done))
```

**Why**:
- Negative experiences need to be more impactful than positive ones
- In RL, avoiding catastrophically bad states is more critical than finding marginally better rewards
- A penalty of -15 becomes -30 in memory, making the Q-value update twice as strong
- Positive rewards unchanged - we want to encourage good behavior but strongly discourage bad

### 5. Episode State Reset

**Added**: `reset_episode()` method called at start of each episode

```python
def reset_episode(self):
    """Reset episode-specific tracking (call at start of each episode)."""
    self.recent_actions.clear()
```

**Why**: Oscillation detection should be episode-specific. An action that causes oscillation in one episode might be valid in a different state configuration.

## Expected Improvements

With these fixes, we expect to see:

1. **No More Infinite Loops**: Oscillation detection will force agent to break out of 2-action traps after 8 repetitions

2. **Faster Learning**: 3x higher learning rate + 2x penalty scaling = 6x stronger learning signal for negative experiences

3. **Better Exploitation**: Faster epsilon decay means agent will exploit learned Q-values by episode 100 instead of episode 500

4. **Decreasing Step Counts**: As agent learns, episode step counts should trend downward toward optimal 7 steps for 3 discs

5. **Varied Action Selection**: Agent should try different strategies when oscillation is detected, leading to discovering better paths

## Monitoring Training

Watch for these indicators in training logs:

### Good Signs
- `⚠️ OSCILLATION DETECTED` messages followed by different actions
- Step counts decreasing over episodes (e.g., 500 → 300 → 150 → 50 → 20 → 10)
- Epsilon decreasing: 0.50 → 0.40 → 0.30 → 0.20 → 0.10 → 0.01
- More variety in penalty types (not just -10.1 and -15.1 repeatedly)
- Occasional completion rewards (+50 to +300)

### Bad Signs (if still occurring)
- Same 2 actions repeating >20 times even with oscillation detection
- Step counts remaining at 500-1000 after 50+ episodes
- Epsilon stuck above 0.2 after 200+ episodes
- No oscillation detection messages at all (might indicate detector not triggering)

## Testing the Fixes

Run training with:
```bash
./venv/bin/python gui_launcher.py
```

Or headless:
```bash
./venv/bin/python train.py 2>&1 | tee training_output.log
```

Monitor for:
1. Oscillation detection messages appearing
2. Actions becoming more diverse after detection
3. Episode completion times decreasing
4. Success rate improving over time

## Further Improvements (if needed)

If oscillation still occurs after these fixes:

1. **Increase penalty_scale to 3.0 or 4.0**: Make bad actions even more undesirable
2. **Reduce oscillation_threshold to 6**: Detect and block oscillations faster (after 3 repetitions instead of 4)
3. **Implement state-based oscillation detection**: Track if agent returns to previously visited states
4. **Add curriculum learning**: Start with 2 discs, then increase to 3 once agent masters simpler problem
5. **Use prioritized experience replay**: Sample experiences with high penalties more frequently during training

## Technical Details

### Q-Value Update Formula
```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

Where:
- α = learning_rate = 0.003 (increased)
- r = reward (scaled by 2.0 if negative)
- γ = gamma = 0.90 (unchanged)

### Why Penalty Scaling Works
If agent receives penalty r=-15 for oscillating:
- Old: Q-value decreases by `0.001 × [-15 + 0.9×Q_next - Q_current]`
- New: Q-value decreases by `0.003 × [-30 + 0.9×Q_next - Q_current]`

The update magnitude is 6x larger (3x from learning rate, 2x from penalty scaling), making the agent learn to avoid oscillating moves much faster.

### Oscillation Detection Algorithm
1. Maintain sliding window of last 20 actions
2. When window has ≥8 actions, check pattern
3. Extract last 8 actions: [a1, a2, a3, a4, a5, a6, a7, a8]
4. If only 2 unique actions and strictly alternating → OSCILLATION
5. Block both actions by setting Q(s, blocked_action) = -∞
6. Agent forced to choose different action
7. Reset detection at start of new episode

This is a **reactive** approach - we detect oscillation after it starts, then break it. Combined with stronger penalties through scaling, the agent should learn to avoid oscillations proactively over time.

## Summary

The core issue was **the agent wasn't learning from penalties effectively**. We fixed this by:
1. ✅ Directly detecting and preventing oscillations at the agent level
2. ✅ Amplifying the learning signal from negative rewards (2x scaling)
3. ✅ Increasing learning rate for faster Q-value updates (3x faster)
4. ✅ Allowing faster transition to exploitation (epsilon decay 5x faster)

These changes create a **6x stronger learning signal for bad actions** while **actively preventing** the specific failure mode (oscillation) that was trapping the agent.
