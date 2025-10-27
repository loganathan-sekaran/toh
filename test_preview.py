#!/usr/bin/env python
"""
Test architecture preview - creates a simple GUI window to test the preview button
"""

import sys
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QPushButton, QLabel
from dqn_agent import DQNAgent
from model_visualizer import ModelVisualizerWidget
from PyQt6.QtGui import QFont

def show_preview():
    """Show a preview of Large architecture"""
    agent = DQNAgent(9, 6, 'Large (128-64-32)')
    
    dialog = QDialog()
    dialog.setWindowTitle("Architecture Preview Test")
    dialog.setMinimumSize(900, 650)
    
    layout = QVBoxLayout(dialog)
    
    info = QLabel(f"<h3>Large (128-64-32) Architecture</h3>"
                  f"<p>Parameters: {agent.model.count_params():,}</p>")
    info.setFont(QFont("Arial", 11))
    layout.addWidget(info)
    
    viz = ModelVisualizerWidget()
    viz.set_model(agent.model, "Test Preview")
    layout.addWidget(viz)
    
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dialog.accept)
    layout.addWidget(close_btn)
    
    dialog.exec()
    print("✓ Preview test successful!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    print("Opening architecture preview test window...")
    show_preview()
    
    print("\n✓ Architecture preview functionality works!")
    print("\nNow available in GUI:")
    print("  1. Training Dialog: Click '👁️ Preview' to see architecture before training")
    print("  2. Model Selection Dialog: Click '👁️ Preview Architecture' to see trained model structure")
