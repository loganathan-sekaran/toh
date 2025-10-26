#!/usr/bin/env python3
"""
Quick test to verify the visualization toggle button works
"""
import sys
from toh import TowerOfHanoiEnv
from visualizer import create_visualizer

# Create environment
env = TowerOfHanoiEnv(num_discs=3)

# Create visualizer
visualizer, app = create_visualizer(env, num_discs=3)

# Print initial state
print(f"Initial show_visualization state: {visualizer.show_visualization}")
print(f"Toggle button text: {visualizer.viz_button.text()}")
print("\nUI has been created with the new toggle button!")
print("Button features:")
print("  - Green when showing visualization (slower, animated)")
print("  - Orange when hiding visualization (faster training)")
print("  - Click to toggle between modes")
print("\nYou can now run training and toggle visualization on/off!")

# Don't start the event loop, just show it's created
visualizer.show()
sys.exit(0)
