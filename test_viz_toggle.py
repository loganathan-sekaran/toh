"""
Quick test for visualization toggle functionality.
"""
import sys
from PyQt6.QtWidgets import QApplication
from toh import TowerOfHanoiEnv
from visualizer import TowerOfHanoiVisualizer

def test_toggle():
    """Test the visualization toggle button."""
    app = QApplication(sys.argv)
    
    env = TowerOfHanoiEnv(num_discs=3)
    viz = TowerOfHanoiVisualizer(env, num_discs=3)
    
    # Set initial state
    env.state = [[3, 2], [1], []]
    viz.canvas.set_state(env.state)
    
    print("Visualizer created successfully")
    print(f"Initial show_visualization: {viz.show_visualization}")
    print(f"Button text: {viz.viz_button.text()}")
    print("\nClick the 'Hide Visualization' button, then 'Show Visualization' to test.")
    print("The canvas should update and show the current state when you click Show.")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    test_toggle()
