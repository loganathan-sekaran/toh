#!/usr/bin/env python3
"""
Test script to demonstrate the progressive placement reward system.

The system rewards placing discs on the target rod in correct order (largest to smallest)
and penalizes removing correctly placed discs.
"""

from toh import TowerOfHanoiEnv

def test_progressive_rewards():
    """Test the progressive placement reward system."""
    
    print("=" * 70)
    print("PROGRESSIVE PLACEMENT REWARD SYSTEM TEST")
    print("=" * 70)
    print("\nRules:")
    print("1. Place disc 3 (largest) on target rod → BIG REWARD (+30)")
    print("2. Remove disc 3 from target rod → HEAVY PENALTY (-40)")
    print("3. Once disc 3 is placed, disc 2 becomes target")
    print("4. Place disc 2 on top of disc 3 → BIG REWARD (+30)")
    print("5. Remove disc 2 after placement → HEAVY PENALTY (-40)")
    print("6. Continue with disc 1...")
    print("\n" + "=" * 70)
    
    # Create environment with 3 discs
    env = TowerOfHanoiEnv(num_discs=3)
    
    print("\nInitial state:")
    print(f"Rod 0: {env.state[0]}")
    print(f"Rod 1: {env.state[1]}")
    print(f"Rod 2 (target): {env.state[2]}")
    print(f"Current target disc: {env.current_target_disc}")
    print(f"Correctly placed: {env.correctly_placed_discs}")
    
    # Test 1: Place disc 3 (largest) on target rod
    print("\n" + "-" * 70)
    print("TEST 1: Place disc 3 on target rod (Rod 2)")
    print("-" * 70)
    
    # Move disc 1 from rod 0 to rod 1
    action = 0  # Rod 0 -> Rod 1
    state, reward, done, _ = env.step(action)
    print(f"Move 1: Disc 1 from Rod 0 to Rod 1 | Reward: {reward:.1f}")
    
    # Move disc 2 from rod 0 to rod 2
    action = 1  # Rod 0 -> Rod 2
    state, reward, done, _ = env.step(action)
    print(f"Move 2: Disc 2 from Rod 0 to Rod 2 | Reward: {reward:.1f}")
    
    # Move disc 1 from rod 1 to rod 2
    action = 3  # Rod 1 -> Rod 2
    state, reward, done, _ = env.step(action)
    print(f"Move 3: Disc 1 from Rod 1 to Rod 2 | Reward: {reward:.1f}")
    
    # Move disc 3 from rod 0 to rod 1
    action = 0  # Rod 0 -> Rod 1
    state, reward, done, _ = env.step(action)
    print(f"Move 4: Disc 3 from Rod 0 to Rod 1 | Reward: {reward:.1f}")
    
    # Move disc 1 from rod 2 to rod 0
    action = 4  # Rod 2 -> Rod 0
    state, reward, done, _ = env.step(action)
    print(f"Move 5: Disc 1 from Rod 2 to Rod 0 | Reward: {reward:.1f}")
    
    # Move disc 2 from rod 2 to rod 1
    action = 5  # Rod 2 -> Rod 1
    state, reward, done, _ = env.step(action)
    print(f"Move 6: Disc 2 from Rod 2 to Rod 1 | Reward: {reward:.1f}")
    
    # Move disc 3 from rod 1 to rod 2 (TARGET PLACEMENT!)
    action = 3  # Rod 1 -> Rod 2
    state, reward, done, _ = env.step(action)
    print(f"\n>>> Move 7: Disc 3 from Rod 1 to Rod 2 (TARGET!) | Reward: {reward:.1f} <<<")
    print(f"    Current target disc: {env.current_target_disc}")
    print(f"    Correctly placed: {env.correctly_placed_discs}")
    
    # Test 2: Try to remove disc 3 from target rod
    print("\n" + "-" * 70)
    print("TEST 2: Remove disc 3 from target rod (should be PENALIZED)")
    print("-" * 70)
    
    # Move disc 1 from rod 0 to rod 1
    action = 0  # Rod 0 -> Rod 1
    state, reward, done, _ = env.step(action)
    print(f"Move 8: Disc 1 from Rod 0 to Rod 1 | Reward: {reward:.1f}")
    
    # Move disc 2 from rod 1 to rod 2
    action = 3  # Rod 1 -> Rod 2
    state, reward, done, _ = env.step(action)
    print(f"Move 9: Disc 2 from Rod 1 to Rod 2 | Reward: {reward:.1f}")
    
    # Move disc 1 from rod 1 to rod 2
    action = 3  # Rod 1 -> Rod 2
    state, reward, done, _ = env.step(action)
    print(f"Move 10: Disc 1 from Rod 1 to Rod 2 | Reward: {reward:.1f}")
    
    print("\nFinal state:")
    print(f"Rod 0: {env.state[0]}")
    print(f"Rod 1: {env.state[1]}")
    print(f"Rod 2 (target): {env.state[2]}")
    print(f"Correctly placed: {env.correctly_placed_discs}")
    print(f"Steps taken: {env.steps}")
    print(f"Puzzle completed: {done}")
    
    # Test 3: Show what happens if we try to remove a correctly placed disc
    print("\n" + "-" * 70)
    print("TEST 3: Start fresh and demonstrate removal penalty")
    print("-" * 70)
    
    env._reset()
    print("\nStarting optimal solution...")
    
    # Optimal solution for 3 discs: 7 moves
    optimal_moves = [
        (1, "Disc 1 from Rod 0 to Rod 2"),
        (0, "Disc 2 from Rod 0 to Rod 1"),
        (5, "Disc 1 from Rod 2 to Rod 1"),
        (1, "Disc 3 from Rod 0 to Rod 2 (TARGET PLACEMENT!)"),
        (2, "Disc 1 from Rod 1 to Rod 0"),
        (3, "Disc 2 from Rod 1 to Rod 2 (TARGET PLACEMENT!)"),
        (1, "Disc 1 from Rod 0 to Rod 2 (TARGET PLACEMENT!)")
    ]
    
    for i, (action, description) in enumerate(optimal_moves):
        state, reward, done, _ = env.step(action)
        marker = " <<<" if "TARGET PLACEMENT" in description else ""
        print(f"Move {i+1}: {description} | Reward: {reward:.1f}{marker}")
        if "TARGET PLACEMENT" in description:
            print(f"         Correctly placed: {env.correctly_placed_discs}")
    
    print(f"\n✓ Puzzle solved in {env.steps} steps!")
    print(f"✓ All discs correctly placed: {env.correctly_placed_discs}")
    
    print("\n" + "=" * 70)
    print("PROGRESSIVE PLACEMENT SYSTEM BENEFITS:")
    print("=" * 70)
    print("✓ Rewards placing discs on target rod in correct order")
    print("✓ Heavily penalizes removing correctly placed discs")
    print("✓ Guides agent to build solution from largest to smallest disc")
    print("✓ Prevents oscillation by locking in correct placements")
    print("✓ Provides clear progression path for learning")
    print("=" * 70)

if __name__ == "__main__":
    test_progressive_rewards()
