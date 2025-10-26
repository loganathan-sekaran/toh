#!/usr/bin/env python3
"""
Quick start script for Tower of Hanoi RL Training.
Run this to see a demo and start training immediately.
"""
import subprocess
import sys
import os


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*80)
    print("  🗼  Tower of Hanoi - Reinforcement Learning Trainer  🗼")
    print("="*80)
    print("\nWelcome! This tool will help you train an AI agent to solve Tower of Hanoi.")
    print("\nWhat would you like to do?\n")


def print_menu():
    """Print menu options."""
    print("1. 🎬 Demo - Watch the optimal solution")
    print("2. 🏋️  Train - Train a new AI model")
    print("3. 🧪 Test - Test a trained model")
    print("4. ⚡ Quick Train - Fast training (500 episodes)")
    print("5. 🎓 Tutorial - Learn how it works")
    print("6. ❌ Exit")
    print()


def run_demo():
    """Run the demo."""
    print("\n" + "="*80)
    print("Running demo with optimal solution...")
    print("="*80 + "\n")
    subprocess.run([sys.executable, "main.py", "demo"])


def run_train(quick=False):
    """Run training."""
    print("\n" + "="*80)
    if quick:
        print("Starting quick training (500 episodes)...")
        print("="*80 + "\n")
        subprocess.run([sys.executable, "main.py", "train", "--episodes", "500", "--show-every", "5"])
    else:
        print("Starting full training (1000 episodes)...")
        print("="*80 + "\n")
        subprocess.run([sys.executable, "main.py", "train"])


def run_test():
    """Run testing."""
    print("\n" + "="*80)
    print("Testing trained model...")
    print("="*80 + "\n")
    
    # Check if models exist
    if not os.path.exists("models") or not os.listdir("models"):
        print("⚠️  No trained models found!")
        print("Please train a model first (option 2 or 4).\n")
        input("Press Enter to continue...")
        return
    
    subprocess.run([sys.executable, "main.py", "test"])


def show_tutorial():
    """Show tutorial information."""
    print("\n" + "="*80)
    print("  📚  Tutorial: How Tower of Hanoi RL Works")
    print("="*80 + "\n")
    
    print("""
🎯 THE PUZZLE:
   - Move all discs from the first rod to the last rod
   - Only one disc can be moved at a time
   - A larger disc cannot be placed on a smaller disc
   - For 3 discs, the optimal solution takes 7 moves (2^3 - 1)

🤖 THE AI AGENT:
   - Uses Deep Q-Network (DQN) - a neural network that learns Q-values
   - Starts by exploring randomly (high epsilon)
   - Gradually learns which moves are good (reward +100 for winning)
   - Gets smarter over time as epsilon decreases

📊 TRAINING PROCESS:
   1. Agent tries random moves and learns from mistakes
   2. Success rate starts low (0-10%) in first 100 episodes
   3. Improves to 50-80% around episode 300-500
   4. Should reach 80%+ success rate after 500-1000 episodes
   5. Best models are automatically saved

🎮 VISUALIZATION:
   - Watch the training in real-time
   - See metrics: episode, steps, rewards, success rate
   - Adjust animation speed with the slider
   - Every 10th episode is visualized by default

💾 MODEL MANAGEMENT:
   - Models are automatically saved every 100 episodes
   - Best model is tracked based on success rate
   - If performance degrades, best model is automatically reloaded
   - All models are saved in the 'models/' directory

⚙️ TIPS FOR SUCCESS:
   ✓ Start with demo to understand the puzzle
   ✓ Train for at least 500 episodes for good results
   ✓ Monitor the success rate - aim for 80%+
   ✓ Test your trained model to see how well it performs
   ✓ For 4 discs, train for 2000+ episodes
    """)
    
    input("\nPress Enter to continue...")


def main():
    """Main menu loop."""
    while True:
        print_banner()
        print_menu()
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == "1":
                run_demo()
            elif choice == "2":
                run_train(quick=False)
            elif choice == "3":
                run_test()
            elif choice == "4":
                run_train(quick=True)
            elif choice == "5":
                show_tutorial()
            elif choice == "6":
                print("\n👋 Thanks for using Tower of Hanoi RL Trainer!")
                print("="*80 + "\n")
                sys.exit(0)
            else:
                print("\n❌ Invalid choice. Please enter 1-6.\n")
                input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using Tower of Hanoi RL Trainer!")
            print("="*80 + "\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
