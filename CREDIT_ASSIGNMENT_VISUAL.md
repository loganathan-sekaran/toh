# Visual Explanation: Temporal Credit Assignment

## The Problem: Agent Can't Learn Which Early Action Caused Later Penalty

### Example Scenario:

```
┌─────────────────────────────────────────────────────────────┐
│  Episode Timeline (Agent trying to solve 3-disc puzzle)     │
└─────────────────────────────────────────────────────────────┘

Step 1:  [[3,2,1],[],[]]  → Move 1→2  → Reward: -0.1
         ⬇
Step 2:  [[3,2],[],[1]]   → Move 0→1  → Reward: -0.1
         ⬇
Step 3:  [[3],[2],[1]]    → Move 2→1  → Reward: -5.0  (reversed!)
         ⬇
Step 4:  [[3],[2,1],[]]   → Move 0→2  → Reward: -0.1
         ⬇
Step 5:  [[],[2,1],[3]]   → Move 1→2  → Reward: -0.1
         ⬇
Step 6:  [[],[2],[3,1]]   → Move 1→0  → Reward: -0.1
         ⬇
Step 7:  [[2],[],[3,1]]   → Move 0→1  → Reward: -10.0 (invalid!)
         ⬇
Step 8:  [[2],[],[3,1]]   → Move 2→1  → Reward: -5.0  (reversed!)
         ⬇
Step 9:  [[2],[1],[3]]    → Move 1→2  → Reward: -25.0 (LOOP!)
                            ❌ OSCILLATION DETECTED
```

---

## OLD System: Only Last Action Gets Penalty

```
Memory Storage (OLD):

┌──────────────────────────────────────────────┐
│ Step 9: (state, action, -25.0, next, done)  │ ← Only this stored
└──────────────────────────────────────────────┘

Problem: Agent learns "avoid THIS state"
But doesn't learn "the sequence starting at Step 6 was bad"
```

**Result**: Agent keeps repeating same mistake sequences! 🔄

---

## NEW System: Credit Assignment Propagates Backwards

```
Memory Storage (NEW):

┌────────────────────────────────────────────────────────────┐
│ Step 9: (state, action, -75.0, next, done)                │ ← Original penalty × 3
│                                                            │
│ 🔗 PROPAGATED BACKWARDS:                                   │
│                                                            │
│ Step 8: (state, action, -42.5, next, done)                │ ← Added -37.5
│         Original: -5.0 → Adjusted: -42.5                  │
│                                                            │
│ Step 7: (state, action, -28.75, next, done)               │ ← Added -18.75
│         Original: -10.0 → Adjusted: -28.75                │
│                                                            │
│ Step 6: (state, action, -9.48, next, done)                │ ← Added -9.38
│         Original: -0.1 → Adjusted: -9.48                  │
│                                                            │
│ Step 5: (state, action, -4.79, next, done)                │ ← Added -4.69
│         Original: -0.1 → Adjusted: -4.79                  │
└────────────────────────────────────────────────────────────┘

Propagation Formula: 
  propagated_penalty = original_penalty × (0.5 ^ distance)
  
  Distance 1 (Step 8): -75 × 0.5¹ = -37.5
  Distance 2 (Step 7): -75 × 0.5² = -18.75
  Distance 3 (Step 6): -75 × 0.5³ = -9.38
  Distance 4 (Step 5): -75 × 0.5⁴ = -4.69
```

**Result**: Agent learns "this whole sequence was bad!" ✅

---

## How Prioritized Replay Helps

### Normal Replay (OLD):

```
Training Batch (64 experiences randomly sampled):

[Good] [Good] [Bad] [Good] [Good] [Good] [Bad] [Good] ...
  ✓      ✓      ✗     ✓      ✓      ✓      ✗     ✓

→ Agent spends 90% time learning "what worked"
→ Only 10% time learning "what failed"
→ Mistakes don't get enough attention!
```

### Prioritized Replay (NEW):

```
Training Batch (64 experiences, stratified):

50% Penalty experiences:   [Bad] [Bad] [Bad] ... (32 experiences)
30% Neutral experiences:   [Meh] [Meh] [Meh] ... (19 experiences)
20% Reward experiences:    [Good] [Good] ... (13 experiences)

→ Agent spends 50% time learning from mistakes
→ Still remembers successful strategies (20%)
→ Balanced learning!
```

---

## Visual: Q-Value Updates

### Before Credit Assignment:

```
Q-Values for State at Step 6: [[2],[],[3,1]]

Actions:        0→1    0→2    1→0    1→2    2→0    2→1
Q-values:      [-5]   [12]   [8]    [15]   [3]    [6]
                                      ⬆
                              Agent picks this
                              (highest Q-value)
                              
Leads to bad outcome at Step 9, but agent doesn't connect it!
```

### After Credit Assignment:

```
Q-Values After Learning from Propagated Experiences:

Actions:        0→1    0→2    1→0    1→2    2→0    2→1
Q-values:      [-5]   [12]   [8]    [3]    [7]    [6]
                                      ⬆       ⬆
                              Now lower!  This becomes better
                              
Agent learns: "Action 1→2 from this state leads to problems"
             "Action 2→0 is actually better"
```

---

## Complete Training Cycle

```
┌─────────────────────────────────────────────────────────┐
│                   EPISODE EXECUTION                      │
└─────────────────────────────────────────────────────────┘
         │
         │ Agent takes actions, gets rewards
         ⬇
┌─────────────────────────────────────────────────────────┐
│              TEMPORAL CREDIT ASSIGNMENT                  │
│                                                          │
│  When penalty occurs:                                   │
│  1. Scale penalty (× 2-3)                               │
│  2. Propagate backwards (5 steps)                       │
│  3. Store all adjusted experiences                      │
└─────────────────────────────────────────────────────────┘
         │
         ⬇
┌─────────────────────────────────────────────────────────┐
│                 EXPERIENCE MEMORY                        │
│                                                          │
│  [Good experiences: 20%] ──────────┐                    │
│  [Neutral experiences: 30%] ───────┤                    │
│  [Penalty experiences: 50%] ───────┘                    │
└─────────────────────────────────────────────────────────┘
         │
         ⬇
┌─────────────────────────────────────────────────────────┐
│              PRIORITIZED REPLAY                          │
│                                                          │
│  Sample batch:                                          │
│  - 50% from penalties (learn from mistakes)             │
│  - 30% from neutral                                     │
│  - 20% from rewards (remember successes)                │
└─────────────────────────────────────────────────────────┘
         │
         ⬇
┌─────────────────────────────────────────────────────────┐
│                Q-VALUE UPDATES                           │
│                                                          │
│  Neural network learns:                                 │
│  "State X + Action Y → leads to penalty sequence"       │
│  → Lower Q-value for that (state, action) pair         │
└─────────────────────────────────────────────────────────┘
         │
         ⬇
┌─────────────────────────────────────────────────────────┐
│              IMPROVED BEHAVIOR                           │
│                                                          │
│  Next episode: Agent avoids similar action sequences    │
│  → Fewer oscillations                                   │
│  → Faster convergence                                   │
└─────────────────────────────────────────────────────────┘
```

---

## Key Insight

**Without Credit Assignment**:
```
Agent thinks: "State S9 is bad" 
              (but doesn't know WHY it got there)
```

**With Credit Assignment**:
```
Agent thinks: "State S9 is bad, AND
               Action at S8 contributed to it, AND
               Action at S7 contributed to it, AND
               Action at S6 started this bad sequence"
```

This is **temporal reasoning** - understanding cause and effect over time! 🧠

---

## Real Training Output You'll See

```
=== EPISODE 245 ===

Step 15:
  State: [[3], [2], [1]]
  Action: 2 (2→1)
  Reward: -0.1, Total: 12.3

Step 16:
  State: [[3], [2,1], []]
  Action: 0 (0→2)
  Reward: -0.1, Total: 12.2

Step 17:
  State: [[], [2,1], [3]]
  Action: 1 (1→0)
  Reward: -5.0, Total: 7.2
  ⚠️ Suboptimal move (penalty: -5.0)

Step 18:
  State: [[2], [1], [3]]
  Action: 0 (0→1)
  Reward: -10.0, Total: -2.8
  ⚠️ Suboptimal move (penalty: -10.0)

Step 19:
  State: [[], [2,1], [3]]
  Action: 1 (1→2)
  Reward: -25.0, Total: -27.8
  ⚠️ Suboptimal move (penalty: -25.0)
  🔗 Credit assignment: Propagated -37.5 penalty to previous action
  🔗 Credit assignment: Propagated -18.8 penalty to 2 steps back
  
Episode 245: Steps=42, Success=No, Reward=-15.2
```

This shows the system **actively learning from the mistake pattern**! 📈
