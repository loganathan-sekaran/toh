#!/usr/bin/env python3
"""
Simple demonstration of the progressive placement reward system.
Shows rewards and penalties for placement/removal actions.
"""

from toh import TowerOfHanoiEnv

def demonstrate_progressive_rewards():
    """Demonstrate the progressive placement reward system step by step."""
    
    print("=" * 80)
    print("PROGRESSIVE PLACEMENT REWARD SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("\n3-Disc Tower of Hanoi - Target: Move all discs to Rod 2")
    print("\nKey Rewards:")
    print("  ✓ Place target disc correctly:  +30 points")
    print("  ✗ Remove correctly placed disc: -40 points")
    print("  + Maintain correct discs:       +1 point per disc")
    print("\n" + "=" * 80)
    
    # Create environment
    env = TowerOfHanoiEnv(num_discs=3)
    
    print("\nInitial State:")
    print(f"  Rod 0: {env.state[0]}")
    print(f"  Rod 1: {env.state[1]}")
    print(f"  Rod 2 (target): {env.state[2]}")
    print(f"  Current target disc: {env.current_target_disc}")
    
    # Execute optimal solution with annotations
    print("\n" + "=" * 80)
    print("OPTIMAL SOLUTION - 7 MOVES")
    print("=" * 80)
    
    moves = [
        (1, "Move disc 1 from Rod 0 to Rod 2"),
        (0, "Move disc 2 from Rod 0 to Rod 1"),
        (5, "Move disc 1 from Rod 2 to Rod 1"),
        (1, "Move disc 3 from Rod 0 to Rod 2 ← PLACE TARGET DISC 3!"),
        (2, "Move disc 1 from Rod 1 to Rod 0"),
        (3, "Move disc 2 from Rod 1 to Rod 2 ← PLACE TARGET DISC 2!"),
        (1, "Move disc 1 from Rod 0 to Rod 2 ← PLACE TARGET DISC 1!")
    ]
    
    for i, (action, description) in enumerate(moves, 1):
        prev_placed = env.correctly_placed_discs.copy()
        prev_target = env.current_target_disc
        
        state, reward, done, _ = env.step(action)
        
        print(f"\nMove {i}: {description}")
        print(f"  Reward: {reward:7.1f}")
        print(f"  Rod 0: {env.state[0]}")
        print(f"  Rod 1: {env.state[1]}")
        print(f"  Rod 2: {env.state[2]}")
        
        if len(env.correctly_placed_discs) > len(prev_placed):
            new_placed = env.correctly_placed_discs - prev_placed
            print(f"  >>> Correctly placed disc {list(new_placed)[0]}! Target now: {env.current_target_disc}")
        
        print(f"  Correctly placed: {sorted(env.correctly_placed_discs, reverse=True)}")
        
        if done:
            print(f"\n  🎉 PUZZLE SOLVED IN {env.steps} STEPS! 🎉")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION: REMOVING CORRECTLY PLACED DISC")
    print("=" * 80)
    
    # Reset and show what happens when removing a correctly placed disc
    env._reset()
    print("\nStarting fresh...")
    
    # Place disc 3 correctly
    for action in [1, 0, 5, 1]:  # Moves to get disc 3 on Rod 2
        env.step(action)
    
    print(f"\nAfter placing disc 3:")
    print(f"  Rod 0: {env.state[0]}")
    print(f"  Rod 1: {env.state[1]}")
    print(f"  Rod 2: {env.state[2]}")
    print(f"  Correctly placed: {sorted(env.correctly_placed_discs, reverse=True)}")
    
    # Now try to remove it
    print("\nNow attempting to place disc 2 on top of disc 3...")
    state, reward1, done, _ = env.step(3)  # Disc 2 to Rod 2
    print(f"  Move: Disc 2 to Rod 2")
    print(f"  Reward: {reward1:7.1f}")
    print(f"  Correctly placed: {sorted(env.correctly_placed_discs, reverse=True)}")
    
    print("\nNow removing disc 2 (correctly placed disc) from Rod 2...")
    state, reward2, done, _ = env.step(5)  # Disc 2 from Rod 2 to Rod 1
    print(f"  Move: Disc 2 from Rod 2 to Rod 1")
    print(f"  Reward: {reward2:7.1f} ← HEAVY PENALTY FOR REMOVAL!")
    print(f"  Correctly placed: {sorted(env.correctly_placed_discs, reverse=True)}")
    print(f"  >>> Disc 2 removed from correctly placed set!")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n✓ Progressive placement system rewards building solution from largest to smallest")
    print("✓ Heavy penalties prevent removing correctly placed discs")
    print("✓ Clear progression path: 3 → 2 → 1")
    print("✓ Reduces oscillation and guides learning")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    demonstrate_progressive_rewards()
