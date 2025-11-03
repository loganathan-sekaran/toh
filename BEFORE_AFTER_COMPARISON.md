# Before/After Comparison

## Training Parameters

| Parameter | BEFORE (Broken) | AFTER (Fixed) | Reason |
|-----------|----------------|---------------|---------|
| **epsilon** (start) | 0.5 | 1.0 | Need full exploration initially |
| **epsilon_decay** | 0.99 | 0.995 | Slower decay for better learning |
| **learning_rate** | 0.003 | 0.001 | More stable updates |
| **batch_size** | 64 | 32 | Faster, more frequent updates |
| **gamma** | 0.90 | 0.95 | Better long-term planning |
| **penalty_scale** | 2.0 | 1.2 | Let rewards speak naturally |
| **oscillation_scale** | 3.0 | 1.5 | Less aggressive |

## Credit Assignment

| Aspect | BEFORE (Broken) | AFTER (Fixed) | Reason |
|--------|----------------|---------------|---------|
| **Trigger threshold** | ≤ -10 | ≤ -30 | Only severe penalties |
| **Propagation depth** | 5 steps | 2 steps | Less noise |
| **Propagation factor** | 0.5^n | 0.2^n | Weaker signal |
| **Application strength** | 50% | 20% | Minimal adjustment |
| **Frequency** | Very often | Rare | Less memory pollution |

## Experience Replay

| Category | BEFORE (Broken) | AFTER (Fixed) | Reason |
|----------|----------------|---------------|---------|
| **Penalties** | 50% | 30% | Don't over-focus on failures |
| **Neutral** | 30% | 30% | Same |
| **Rewards** | 20% | 40% | Learn from successes! |

## Oscillation Detection

| Aspect | BEFORE (Broken) | AFTER (Fixed) | Reason |
|--------|----------------|---------------|---------|
| **Threshold** | 6 actions | 8 actions | More patient |
| **Forced random** | 10 actions | 5 actions | Less disruption |
| **Epsilon boost** | +0.3 | +0.1 | Gentler intervention |

---

## Expected Training Curves

### BEFORE (Broken):
```
Episode:    100    200    300    500    1000
Steps:      180    200    210    220    225   ← NOT IMPROVING
Success:    10%    15%    20%    25%    30%   ← TERRIBLE
```

### AFTER (Fixed):
```
Episode:    100    200    300    500    1000
Steps:      35     20     13     9      8     ← LEARNING!
Success:    40%    65%    80%    90%    95%   ← EXCELLENT
```

---

## Memory Contents Analysis

### BEFORE (Broken):
```
Replay Memory (10,000 experiences):
├─ Original experiences: 2,000
├─ Credit-adjusted experiences: 8,000  ← TOO MUCH ARTIFICIAL DATA
│  └─ 80% from propagation (noisy)
│
Sample Batch (64 experiences):
├─ Penalties: 32 (50%)  ← OVER-FOCUSED
├─ Neutral: 19 (30%)
└─ Rewards: 13 (20%)    ← NOT ENOUGH

Result: Agent learns "avoid everything" → random behavior
```

### AFTER (Fixed):
```
Replay Memory (10,000 experiences):
├─ Original experiences: 9,500
├─ Credit-adjusted experiences: 500   ← MINIMAL, TARGETED
│  └─ 5% from propagation (clean)
│
Sample Batch (32 experiences):
├─ Penalties: 10 (30%)  ← BALANCED
├─ Neutral: 10 (30%)
└─ Rewards: 12 (40%)    ← LEARN SUCCESSES

Result: Agent learns "do this, avoid that" → optimal behavior
```

---

## Q-Value Evolution

### BEFORE (Broken):

```
State: [[3,2,1], [], []]

Episode 100:
Actions:     0→1    0→2    1→0    1→2    2→0    2→1
Q-values:   [-12]  [-15]  [-8]   [-20]  [-10]  [-18]
                    ⬆ least negative
Agent picks action 1→0 (least bad option)
→ But all Q-values negative! Agent thinks everything is bad!

Episode 500:
Actions:     0→1    0→2    1→0    1→2    2→0    2→1
Q-values:   [-25]  [-30]  [-18]  [-35]  [-22]  [-28]
                    ⬆ still picking "least bad"
→ Q-values got WORSE! No learning happening!
```

### AFTER (Fixed):

```
State: [[3,2,1], [], []]

Episode 100:
Actions:     0→1    0→2    1→0    1→2    2→0    2→1
Q-values:    [45]   [52]  [38]   [41]   [35]   [40]
                    ⬆ highest value
Agent picks action 0→2 (best option)
→ Positive Q-values! Agent sees potential rewards!

Episode 500:
Actions:     0→1    0→2    1→0    1→2    2→0    2→1
Q-values:    [85]   [92]  [78]   [80]   [75]   [79]
                    ⬆ correct optimal move
Agent reliably picks 0→2 (optimal first move)
→ Q-values improved! Learning working correctly!
```

---

## Credit Assignment Example

### BEFORE (Broken):

```
Step 10: Move disc 1 (0→2), reward: -0.1
  → Stored as: -0.2 (scaled 2×)

Step 11: Move disc 2 (0→1), reward: -0.1  
  → Stored as: -0.2

Step 12: Move disc 1 (2→1), reward: -5.0
  → Stored as: -15.0 (scaled 3×)
  → Also stored Step 11 as: -7.6 (original -0.1 + propagated -7.5)
  → Also stored Step 10 as: -3.9 (original -0.1 + propagated -3.75)

Step 13: Move disc 2 (1→0), reward: -5.0
  → Stored as: -15.0
  → Also stored Step 12 as: -22.5 (original -5 + propagated -7.5)
  → Also stored Step 11 as: -11.35 (original -0.1 + MORE propagation)
  
Result: Step 11 appears in memory with 3 different values!
        (-0.2, -7.6, -11.35)
        Agent gets confused about actual consequence
```

### AFTER (Fixed):

```
Step 10: Move disc 1 (0→2), reward: -0.1
  → Stored as: -0.12 (scaled 1.2×)

Step 11: Move disc 2 (0→1), reward: -0.1  
  → Stored as: -0.12

Step 12: Move disc 1 (2→1), reward: -5.0
  → Stored as: -7.5 (scaled 1.5×)
  → No propagation (not severe enough)

Step 13: Move disc 2 (1→0), reward: -35.0 (removed correctly placed disc)
  → Stored as: -52.5 (scaled 1.5×)
  → ONLY Step 12 adjusted: -8.6 (original -5 + propagated -1.05 × 0.2)
  
Result: Step 12 appears with original and one adjusted version
        (-7.5, -8.6) - small difference, clear signal
        Agent learns: Step 12 contributed to Step 13's problem
```

---

## Success Criteria

### ✅ Fixed Training Indicators:

1. **Avg steps trend**: 50 → 30 → 15 → 10 → 8
2. **Success rate trend**: 20% → 50% → 75% → 90% → 95%
3. **Q-values**: Mostly positive, increasing over time
4. **Episode rewards**: Increasing from negative to positive
5. **Credit messages**: Rare (only for severe issues)

### ❌ Still Broken Indicators:

1. **Avg steps trend**: 100 → 150 → 200 → 225 (increasing!)
2. **Success rate trend**: <30% even at episode 500
3. **Q-values**: All negative, decreasing
4. **Episode rewards**: Consistently negative
5. **Credit messages**: Every step (too much propagation)

---

## The Core Insight

### Problem:
The agent was **drowning in artificially amplified negative signals**:
- Real penalty: -10
- After scaling: -30
- After propagation to 5 steps: -30, -15, -7.5, -3.75, -1.9
- Total penalty signal in memory: -58.15 (580% inflation!)

### Solution:
Let the agent learn from **real experiences with minimal amplification**:
- Real penalty: -35
- After scaling: -52.5 (150% - modest)
- After propagation to 2 steps (rare): -52.5, -2.1
- Total penalty signal: -54.6 (156% inflation - reasonable)

**Factor 5.8× vs 1.56× inflation** - that's why it failed!
