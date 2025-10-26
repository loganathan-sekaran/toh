#!/usr/bin/env python3
"""
Quick test of all components
"""

print("Testing imports...")
try:
    from PyQt6.QtWidgets import QApplication
    print("✓ PyQt6 imports working")
except ImportError as e:
    print(f"✗ PyQt6 import failed: {e}")
    exit(1)

try:
    from toh import TowerOfHanoiEnv
    print("✓ TowerOfHanoiEnv import working")
except ImportError as e:
    print(f"✗ TowerOfHanoiEnv import failed: {e}")
    exit(1)

try:
    from dqn_agent import DQNAgent
    print("✓ DQNAgent import working")
except ImportError as e:
    print(f"✗ DQNAgent import failed: {e}")
    exit(1)

try:
    from visualizer import create_visualizer
    print("✓ Visualizer import working")
except ImportError as e:
    print(f"✗ Visualizer import failed: {e}")
    exit(1)

try:
    from train_with_gui import TrainingMetrics, ModelManager
    print("✓ Training module imports working")
except ImportError as e:
    print(f"✗ Training module import failed: {e}")
    exit(1)

print("\nTesting environment...")
try:
    env = TowerOfHanoiEnv(num_discs=3)
    state = env._reset()
    print(f"✓ Environment created: {len(env.state[0])} discs on first rod")
    
    # Test a move
    next_state, reward, done, _ = env.step(0)
    print(f"✓ Environment step working: reward={reward}, done={done}")
except Exception as e:
    print(f"✗ Environment test failed: {e}")
    exit(1)

print("\nTesting agent...")
try:
    import numpy as np
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    print(f"✓ Agent created: state_size={state_size}, action_size={action_size}")
    
    # Test action selection
    from util import flatten_state
    flat_state = flatten_state(state, env.num_discs)
    action = agent.act(flat_state)
    print(f"✓ Agent action selection working: action={action}")
except Exception as e:
    print(f"✗ Agent test failed: {e}")
    exit(1)

print("\n" + "="*60)
print("✅ All tests passed!")
print("="*60)
print("\nYou can now:")
print("  • Run demo:  python main.py demo")
print("  • Start training:  python main.py train --episodes 100")
print("  • Interactive menu:  python main.py")
print("  • Or use:  ./start.sh")
