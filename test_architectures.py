#!/usr/bin/env python
"""
Quick test script to demonstrate the modular model architecture system.
Shows how to list, compare, and use different model architectures.
"""

from model_architectures import ModelFactory
from dqn_agent import DQNAgent

def main():
    print("=" * 80)
    print("🎯 Tower of Hanoi - Modular Model Architecture System")
    print("=" * 80)
    
    # List all available architectures
    print("\n📐 Available Model Architectures:\n")
    ModelFactory.list_architectures()
    
    # Compare model sizes
    print("\n📊 Model Size Comparison:\n")
    print(f"{'Architecture':<30} {'Parameters':>15} {'Complexity':<15}")
    print("-" * 60)
    
    for arch_name in ModelFactory.get_architecture_names():
        arch = ModelFactory.get_architecture(arch_name)
        info = arch.get_info()
        
        # Create a sample agent to count parameters
        agent = DQNAgent(9, 6, arch_name)
        param_count = agent.model.count_params()
        
        print(f"{arch_name:<30} {param_count:>15,} {info['complexity']:<15}")
    
    print("\n" + "=" * 80)
    print("✅ All architectures loaded successfully!")
    print("=" * 80)
    print("\n💡 Usage Tips:")
    print("  • Small (24-24): Fast experiments, baseline comparison")
    print("  • Medium (64-32): Balanced performance and speed")
    print("  • Large (128-64-32): Best results, recommended default")
    print("  • Extra Large (256-128-64): Maximum capacity for complex problems")
    print("\n🚀 Start the GUI to select and train different models:")
    print("   ./start_gui.sh")
    print()

if __name__ == '__main__':
    main()
