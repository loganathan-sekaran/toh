# Progressive Placement Reward System

## Overview
The progressive placement reward system encourages the agent to build the solution incrementally from largest disc to smallest, rewarding correct placements and heavily penalizing the removal of correctly placed discs.

## How It Works

### Target Disc Tracking
- Starts with the **largest disc** (disc N) as the current target
- Once a disc is correctly placed on the target rod, the **next smaller disc** becomes the target
- Tracks which discs are correctly placed using `correctly_placed_discs` set

### Placement Rules

#### Disc N (Largest)
- **Reward (+30)**: Place disc N on target rod when it's the **only disc** on target rod
- **Penalty (-40)**: Remove disc N from target rod after it's correctly placed

#### Disc N-1, N-2, ... (Smaller discs)
- **Reward (+30)**: Place disc on target rod **on top of the next larger disc** (disc + 1) that's already correctly placed
- **Penalty (-40)**: Remove a correctly placed disc from target rod
- **Small reward (+1)**: Each step maintaining correctly placed discs on target

### Example: 3-Disc Tower (Optimal Solution)

```
Initial State:
Rod 0: [3, 2, 1]  (largest to smallest)
Rod 1: []
Rod 2: []  (target)
Target disc: 3
```

**Step-by-step rewards:**

1. **Move disc 1 to rod 2**: +7.9 (small progress)
2. **Move disc 2 to rod 1**: -0.1 (base penalty)
3. **Move disc 1 to rod 1**: -0.1 (base penalty)
4. **Move disc 3 to rod 2**: **+83.9** (+30 for correct placement + other bonuses)
   - Target disc is now: 2
   - Correctly placed: {3}
5. **Move disc 1 to rod 0**: +2.9 (maintaining disc 3)
6. **Move disc 2 to rod 2**: **+38.9** (+30 for placement on top of disc 3)
   - Target disc is now: 1
   - Correctly placed: {2, 3}
7. **Move disc 1 to rod 2**: **+300.0** (+30 for placement + completion bonus)
   - Correctly placed: {1, 2, 3}
   - **Puzzle solved in optimal 7 steps!**

## Key Benefits

### 1. **Prevents Oscillation**
Once a disc is correctly placed, removing it incurs a heavy penalty (-40), discouraging the agent from undoing progress.

### 2. **Clear Learning Path**
The progressive nature (largest → smallest) provides a clear sequence of goals, making it easier for the agent to learn the optimal strategy.

### 3. **Incremental Progress**
Each correctly placed disc earns a substantial reward (+30), providing clear feedback that the agent is moving in the right direction.

### 4. **Maintains Valid State**
The system ensures discs are placed in the correct order (descending from bottom to top) on the target rod.

## Integration with Other Reward Components

The progressive placement system works alongside:
- **Invalid move penalty**: -50 for placing larger disc on smaller disc
- **Repetition detection**: Penalties for oscillating between rods
- **Efficiency bonus**: Extra rewards for completing in fewer steps
- **Completion reward**: 50-200 points for solving the puzzle

## Testing

Run the test script to see the system in action:
```bash
python test_progressive_rewards.py
```

This demonstrates:
- Correct placement rewards for each disc
- Penalties for removing correctly placed discs
- The optimal 7-step solution for 3 discs
- Progressive tracking from largest to smallest disc

## Configuration

Key parameters in `toh.py`:
- `correctly_placed_discs`: Set tracking correctly placed disc numbers
- `current_target_disc`: Next disc that should be placed (largest to smallest)
- Placement reward: +30 points
- Removal penalty: -40 points
- Maintenance reward: +1 point per correctly placed disc

## Expected Learning Improvements

With this system, the agent should:
1. Learn to place the largest disc first
2. Avoid moving correctly placed discs
3. Build the solution incrementally
4. Achieve optimal or near-optimal solutions faster
5. Reduce oscillation and repetitive moves
