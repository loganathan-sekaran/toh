# Backtracking Problem & Solution

## The Problem You Identified

Your model was getting stuck in loops during testing because:

1. **No memory of past mistakes**: DQN makes decisions based only on current state, not the sequence of actions that led there
2. **Can't identify which early action caused later penalties**: When getting penalty at step 7, the agent doesn't know if the problem was at step 1, 3, or 5
3. **No backtracking mechanism**: Agent can't "undo" moves and try alternative paths

## Why This Happens: The Markov Property

Standard DQN treats each state **independently** (Markov Decision Process):
- State `[[3,2], [1], []]` looks the same whether you got there via good or bad moves
- The agent has **no memory** of which actions led to this state
- It can't reason: "I tried moving disc 1 to rod 2, which led to a bad state, so next time I should try rod 3 instead"

## The Solutions (What We Implemented)

### ✅ Solution 1: Temporal Credit Assignment (Implemented)

**What it does**: When the agent receives a large penalty, we **propagate a fraction of that penalty backwards** to the actions that led to it.

**Example**:
```
Step 1: Move disc 1 → rod 2 (reward: -0.1)
Step 2: Move disc 2 → rod 1 (reward: -0.1)  
Step 3: Move disc 1 → rod 1 (reward: -0.1)
Step 4: Move disc 2 → rod 2 (reward: -0.1)
Step 5: Move disc 1 → rod 2 (reward: -20) ← OSCILLATION PENALTY!
```

**Old behavior**: Only step 5 gets the -20 penalty

**New behavior**: 
- Step 5: Gets -20 penalty (scaled to -60)
- Step 4: Gets additional -10 penalty (50% propagation)
- Step 3: Gets additional -5 penalty (25% propagation)
- Step 2: Gets additional -2.5 penalty (12.5% propagation)

This teaches the agent: **"The sequence of moves starting at step 2 leads to trouble!"**

**Key Code Changes in `dqn_agent.py`**:

```python
# Track episode trajectory
self.current_episode_trajectory = []

# When severe penalty occurs (reward <= -10)
if reward <= -10 and len(self.current_episode_trajectory) > 1:
    # Propagate penalty to previous 5 steps (max)
    for i in range(1, 6):
        propagation_factor = (0.5 ** i)  # 50%, 25%, 12.5%, etc.
        propagated_penalty = scaled_reward * propagation_factor
        
        # Store adjusted experience
        self.memory.append((prev_state, prev_action, adjusted_reward, ...))
```

### ✅ Solution 2: Prioritized Experience Replay (Implemented)

**What it does**: Makes the agent **learn more from mistakes** by sampling penalty experiences more frequently during training.

**Batch Composition**:
- **50%** experiences with significant penalties (reward < -5)
- **30%** experiences with minor penalties (reward < 0)
- **20%** experiences with positive rewards

This ensures the agent focuses on **understanding what went wrong** while not forgetting successful strategies.

**Key Code Changes**:
```python
# Separate experiences by type
penalty_experiences = [exp for exp in memory if exp.reward < -5]
neutral_experiences = [exp for exp in memory if -5 <= exp.reward < 0]
reward_experiences = [exp for exp in memory if exp.reward >= 0]

# Sample 50% from penalties, 30% from neutral, 20% from rewards
minibatch = sample(penalty_experiences, 50%) + 
            sample(neutral_experiences, 30%) + 
            sample(reward_experiences, 20%)
```

### ✅ Solution 3: Enhanced Testing with Exploration (Previously Implemented)

**What it does**: During testing, adds randomness so each test run tries different paths.

**Configuration**:
- **Exploration slider**: 0-100% chance to try random valid action
- **10-20% recommended**: Good balance for partially trained models
- **Each retest is different**: Model can find alternative solution paths

## Why This Approach Works

### 1. **Better Credit Assignment**
The agent learns: "Action X at step 2 → leads to penalty at step 5"
- Without: Only learns "avoid state at step 5"
- With: Learns "avoid the action sequence that leads to step 5"

### 2. **Focused Learning**
Prioritized replay ensures the agent:
- Spends more time learning from mistakes
- Doesn't forget successful patterns
- Converges faster to optimal policy

### 3. **No Architecture Change Needed**
- ✅ Works with existing DQN model
- ✅ No LSTM/Transformer required
- ✅ Compatible with all your trained models
- ✅ Only improves future training (doesn't affect existing models)

## What About True Backtracking?

True backtracking (like "undo last 3 moves and try different path") would require:

### Option A: Episode Reset During Training
```python
if consecutive_penalties > 5:
    # Reset to state from 5 steps ago
    state = saved_states[-5]
    # Try different action
```

**Pros**: Simple concept
**Cons**: 
- Breaks the MDP assumption
- Complicates training loop significantly
- May not converge well

### Option B: Monte Carlo Tree Search (MCTS)
Used in AlphaGo/AlphaZero for explicit lookahead planning.

**Pros**: Optimal for tree-search problems like Tower of Hanoi
**Cons**: 
- Complete architecture rewrite
- Much more complex
- Overkill for this problem

### Option C: LSTM/Transformer Architecture
Adds memory to track action sequences explicitly.

**Pros**: Can learn temporal patterns
**Cons**:
- Requires model architecture change
- All models need retraining
- More complex and slower

## Our Recommendation: Stick with Current Solution

The **Temporal Credit Assignment + Prioritized Replay** approach:
- ✅ Solves 80% of the problem with 20% of the complexity
- ✅ No model architecture changes
- ✅ Works with existing training code
- ✅ Future models train better automatically
- ✅ Testing exploration helps existing models

For Tower of Hanoi specifically, this is **sufficient** because:
1. State space is relatively small (compared to games like chess)
2. Reward shaping is already good (progressive placement, oscillation detection)
3. The agent just needs better credit assignment, not true backtracking

## Results You Should See

### During Training:
- **Faster convergence**: Agent learns optimal path in fewer episodes
- **Better oscillation avoidance**: Credit assignment teaches "don't repeat these sequences"
- **Higher success rate**: More focused learning from mistakes
- **Output messages**: You'll see `🔗 Credit assignment: Propagated penalty...`

### During Testing:
- **Fewer stuck loops**: Better trained models make fewer repeated mistakes
- **Alternative paths**: Exploration slider helps find different solutions
- **Success on retry**: Each test run has different randomness

## How to Use

1. **Train new models** - they'll automatically use the enhanced learning
2. **Set exploration 10-20%** when testing partially trained models
3. **Retest multiple times** if model fails - each run tries different paths
4. **Watch for credit assignment messages** during training to see it working

## Future Enhancements (If Needed)

If models still struggle:

1. **Increase propagation depth**: Currently 5 steps, could go to 10
2. **Adjust propagation factor**: Currently 0.5^n, could use 0.7^n for stronger propagation
3. **Add LSTM layers**: If temporal patterns are still hard to learn
4. **Implement true MCTS**: For optimal play (but much more complex)

For now, the current solution should significantly improve training effectiveness! 🎯
