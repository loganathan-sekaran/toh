"""
PyQt6-based Tower of Hanoi Visualizer
Interactive GUI with animation and training controls
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QSlider, QFrame)
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont
import numpy as np


class TowerOfHanoiCanvas(QWidget):
    """Custom widget for drawing the Tower of Hanoi state"""
    
    def __init__(self, num_discs=3):
        super().__init__()
        self.num_discs = num_discs
        self.state = [list(range(num_discs, 0, -1)), [], []]
        self.animating = False
        self.animation_progress = 0.0
        self.animation_from_rod = None
        self.animation_to_rod = None
        self.animation_disc = None
        
        self.setMinimumSize(800, 400)
        
        # Color palette
        self.bg_color = QColor(245, 245, 250)
        self.rod_color = QColor(60, 60, 70)
        self.base_color = QColor(80, 80, 90)
        self.disc_colors = [
            QColor(255, 107, 107),  # Red
            QColor(78, 205, 196),   # Teal
            QColor(255, 195, 0),    # Yellow
            QColor(106, 137, 204),  # Blue
            QColor(184, 143, 235),  # Purple
            QColor(72, 219, 156),   # Green
        ]
    
    def set_state(self, new_state):
        """Update the tower state"""
        self.state = [list(rod) for rod in new_state]
        self.update()
    
    def start_animation(self, from_rod, to_rod, disc):
        """Start animating a disc move"""
        self.animating = True
        self.animation_progress = 0.0
        self.animation_from_rod = from_rod
        self.animation_to_rod = to_rod
        self.animation_disc = disc
    
    def update_animation(self, progress):
        """Update animation progress (0.0 to 1.0)"""
        self.animation_progress = progress
        self.update()
    
    def end_animation(self):
        """End the animation"""
        self.animating = False
        self.animation_progress = 0.0
        self.update()
    
    def paintEvent(self, event):
        """Draw the Tower of Hanoi"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self.bg_color)
        
        # Calculate dimensions
        width = self.width()
        height = self.height()
        margin = 50
        rod_spacing = (width - 2 * margin) / 3
        base_y = height - margin
        rod_height = height - 2 * margin - 50
        rod_width = 8
        base_height = 15
        
        # Draw bases
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.base_color))
        for i in range(3):
            rod_x = margin + i * rod_spacing + rod_spacing / 2
            base_rect = QRectF(rod_x - 100, base_y, 200, base_height)
            painter.drawRoundedRect(base_rect, 5, 5)
        
        # Draw rods
        painter.setBrush(QBrush(self.rod_color))
        for i in range(3):
            rod_x = margin + i * rod_spacing + rod_spacing / 2 - rod_width / 2
            rod_rect = QRectF(rod_x, base_y - rod_height, rod_width, rod_height)
            painter.drawRoundedRect(rod_rect, 3, 3)
        
        # Draw rod labels
        painter.setPen(QPen(QColor(60, 60, 70)))
        font = QFont("Arial", 14, QFont.Weight.Bold)
        painter.setFont(font)
        for i, label in enumerate(['A', 'B', 'C']):
            rod_x = margin + i * rod_spacing + rod_spacing / 2
            painter.drawText(int(rod_x - 10), int(base_y + 35), label)
        
        # Calculate disc dimensions
        max_disc_width = 180
        min_disc_width = 40
        disc_height = min(40, rod_height / (self.num_discs + 1))
        
        # Draw discs
        for rod_idx, rod in enumerate(self.state):
            rod_x = margin + rod_idx * rod_spacing + rod_spacing / 2
            
            for pos_idx, disc in enumerate(rod):
                # Skip the disc being animated
                if (self.animating and 
                    rod_idx == self.animation_from_rod and 
                    disc == self.animation_disc):
                    continue
                
                disc_width = min_disc_width + (max_disc_width - min_disc_width) * (disc / self.num_discs)
                disc_y = base_y - (pos_idx + 1) * disc_height - 5
                
                color = self.disc_colors[disc - 1] if disc <= len(self.disc_colors) else self.disc_colors[0]
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color.darker(120), 2))
                
                disc_rect = QRectF(rod_x - disc_width / 2, disc_y, disc_width, disc_height - 2)
                painter.drawRoundedRect(disc_rect, 8, 8)
                
                # Draw disc number
                painter.setPen(QPen(Qt.GlobalColor.white))
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                painter.drawText(disc_rect, Qt.AlignmentFlag.AlignCenter, str(disc))
        
        # Draw animated disc
        if self.animating and self.animation_disc:
            from_x = margin + self.animation_from_rod * rod_spacing + rod_spacing / 2
            to_x = margin + self.animation_to_rod * rod_spacing + rod_spacing / 2
            
            # Calculate animation path (up, across, down)
            if self.animation_progress < 0.33:
                # Rising phase
                t = self.animation_progress / 0.33
                x = from_x
                from_height = len([d for d in self.state[self.animation_from_rod] if d != self.animation_disc])
                start_y = base_y - (from_height + 1) * disc_height - 5
                y = start_y - t * 100
            elif self.animation_progress < 0.67:
                # Moving across phase
                t = (self.animation_progress - 0.33) / 0.34
                x = from_x + (to_x - from_x) * t
                from_height = len([d for d in self.state[self.animation_from_rod] if d != self.animation_disc])
                start_y = base_y - (from_height + 1) * disc_height - 5
                y = start_y - 100
            else:
                # Descending phase
                t = (self.animation_progress - 0.67) / 0.33
                x = to_x
                to_height = len(self.state[self.animation_to_rod])
                end_y = base_y - (to_height + 1) * disc_height - 5
                from_height = len([d for d in self.state[self.animation_from_rod] if d != self.animation_disc])
                start_y = base_y - (from_height + 1) * disc_height - 5
                y = (start_y - 100) + t * 100 + t * (end_y - start_y)
            
            disc_width = min_disc_width + (max_disc_width - min_disc_width) * (self.animation_disc / self.num_discs)
            color = self.disc_colors[self.animation_disc - 1] if self.animation_disc <= len(self.disc_colors) else self.disc_colors[0]
            
            # Draw shadow
            shadow_color = QColor(0, 0, 0, 30)
            painter.setBrush(QBrush(shadow_color))
            painter.setPen(Qt.PenStyle.NoPen)
            shadow_rect = QRectF(x - disc_width / 2 + 5, y + 5, disc_width, disc_height - 2)
            painter.drawRoundedRect(shadow_rect, 8, 8)
            
            # Draw disc
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(120), 2))
            disc_rect = QRectF(x - disc_width / 2, y, disc_width, disc_height - 2)
            painter.drawRoundedRect(disc_rect, 8, 8)
            
            # Draw disc number
            painter.setPen(QPen(Qt.GlobalColor.white))
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            painter.drawText(disc_rect, Qt.AlignmentFlag.AlignCenter, str(self.animation_disc))


class TowerOfHanoiVisualizer(QMainWindow):
    """Main visualizer window with controls"""
    
    def __init__(self, env, num_discs=3, standalone=True):
        super().__init__()
        self.env = env
        self.num_discs = num_discs
        self.animation_speed = 20  # milliseconds per frame (lower = faster)
        self.paused = False
        self.standalone = standalone  # Whether this is a standalone window or embedded
        
        self.setWindowTitle("Tower of Hanoi - RL Training")
        self.setGeometry(100, 100, 1000, 700)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Tower of Hanoi - Reinforcement Learning")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Canvas
        self.canvas = TowerOfHanoiCanvas(num_discs)
        main_layout.addWidget(self.canvas)
        
        # Info panel
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        info_layout = QHBoxLayout(info_frame)
        
        self.episode_label = QLabel("Episode: 0")
        self.step_label = QLabel("Step: 0")
        self.reward_label = QLabel("Reward: 0.0")
        self.epsilon_label = QLabel("Epsilon: 1.0")
        self.success_label = QLabel("Success Rate: 0%")
        
        for label in [self.episode_label, self.step_label, self.reward_label, 
                     self.epsilon_label, self.success_label]:
            label.setFont(QFont("Arial", 11))
            info_layout.addWidget(label)
        
        main_layout.addWidget(info_frame)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Animation Speed:")
        speed_label.setFont(QFont("Arial", 11))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(30)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(20)
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        self.speed_value_label = QLabel("30 ms/frame")
        self.speed_value_label.setFont(QFont("Arial", 11))
        
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_value_label)
        main_layout.addLayout(speed_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.setFont(QFont("Arial", 12))
        self.pause_button.clicked.connect(self.toggle_pause)
        
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setFont(QFont("Arial", 12))
        self.stop_button.clicked.connect(self.stop_training)
        
        self.test_button = QPushButton("Test Model")
        self.test_button.setFont(QFont("Arial", 12))
        self.test_button.clicked.connect(self.test_model)
        
        # Toggle visualization button
        self.viz_button = QPushButton("Hide Visualization (Fast Training)")
        self.viz_button.setFont(QFont("Arial", 12))
        self.viz_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.viz_button.clicked.connect(self.toggle_visualization)
        self.show_visualization = True  # Start with visualization enabled
        
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.viz_button)
        button_layout.addWidget(self.test_button)
        main_layout.addLayout(button_layout)
        
        # Animation frames for smooth movement
        self.animation_frames = 30  # Frames per animation (30 frames @ 20ms = 600ms per move)
        
        # Training control
        self.should_stop = False
        
        # Only show as standalone window if specified
        if self.standalone:
            self.show()
    
    def on_speed_change(self, value):
        """Handle speed slider change"""
        self.animation_speed = value
        self.speed_value_label.setText(f"{value} ms/frame")
    
    def toggle_pause(self):
        """Pause/resume training"""
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")
    
    def toggle_visualization(self):
        """Toggle between showing/hiding visualization for fast training"""
        self.show_visualization = not self.show_visualization
        if self.show_visualization:
            self.viz_button.setText("Hide Visualization (Fast Training)")
            self.viz_button.setStyleSheet("background-color: #4CAF50; color: white;")
            # Immediately show the current state when enabling visualization
            # Use update() instead of repaint() to avoid QPainter errors
            self.canvas.set_state(self.env.state)
            QApplication.processEvents()
            print(f"[Visualization ON] Current state: {self.env.state}")
        else:
            self.viz_button.setText("Show Visualization (Currently Fast)")
            self.viz_button.setStyleSheet("background-color: #FF9800; color: white;")
            print(f"[Visualization OFF] Fast training mode enabled")
            QApplication.processEvents()
    
    def stop_training(self):
        """Stop the training"""
        self.should_stop = True
        self.close()
    
    def test_model(self):
        """Placeholder for testing trained model"""
        print("Test model functionality - to be implemented")
    
    def update_state(self, state):
        """Update the displayed state"""
        self.canvas.set_state(state)
        QApplication.processEvents()
    
    def animate_move(self, from_rod, to_rod, disc):
        """Animate a disc move with smooth frame-by-frame rendering"""
        import time
        
        # If visualization is hidden, skip animation for fast training
        if not self.show_visualization:
            # Just update the final state without animation
            # Don't force immediate repaint - let Qt handle it naturally
            self.canvas.set_state(self.env.state)
            QApplication.processEvents()
            return
        
        # Visualization is ON - perform full animation
        print(f"[Animating] Move disc {disc} from rod {from_rod} to rod {to_rod}")
        self.canvas.start_animation(from_rod, to_rod, disc)
        
        # Render each frame of the animation
        for frame in range(self.animation_frames):
            progress = frame / self.animation_frames
            
            # Update the animation progress
            self.canvas.animation_progress = progress
            self.canvas.update()
            
            # Process Qt events to render the frame
            QApplication.processEvents()
            
            # Sleep for smooth animation
            time.sleep(self.animation_speed / 1000.0)
        
        # End animation with final frame
        self.canvas.end_animation()
        self.canvas.update()
        QApplication.processEvents()
    
    def update_info(self, episode=None, step=None, reward=None, epsilon=None, success_rate=None):
        """Update the information display"""
        # Update metrics labels (always update these for monitoring)
        if episode is not None:
            self.episode_label.setText(f"Episode: {episode}")
        if step is not None:
            self.step_label.setText(f"Step: {step}")
        if reward is not None:
            self.reward_label.setText(f"Reward: {reward:.2f}")
        if epsilon is not None:
            self.epsilon_label.setText(f"Epsilon: {epsilon:.3f}")
        if success_rate is not None:
            self.success_label.setText(f"Success Rate: {success_rate:.1f}%")
        
        # Always process events to keep UI responsive
        # This is lightweight and ensures UI updates even in fast mode
        QApplication.processEvents()
    
    def wait_if_paused(self):
        """Block execution while paused"""
        while self.paused and not self.should_stop:
            QApplication.processEvents()
            QTimer.singleShot(100, lambda: None)


def create_visualizer(env, num_discs=3):
    """Create and return a visualizer instance"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    visualizer = TowerOfHanoiVisualizer(env, num_discs)
    return visualizer, app


if __name__ == "__main__":
    # Demo mode
    from toh import TowerOfHanoiEnv
    
    env = TowerOfHanoiEnv(num_discs=3)
    visualizer, app = create_visualizer(env, num_discs=3)
    
    # Demo animation
    def demo():
        import time
        visualizer.update_state(env.state)
        time.sleep(1)
        
        # Perform some moves
        moves = [(0, 2), (0, 1), (2, 1), (0, 2), (1, 0), (1, 2), (0, 2)]
        for from_rod, to_rod in moves:
            if visualizer.should_stop:
                break
            
            # Find disc to move
            disc = env.state[from_rod][-1] if env.state[from_rod] else None
            if disc:
                # Remove disc from source
                env.state[from_rod].pop()
                
                # Animate
                visualizer.animate_move(from_rod, to_rod, disc)
                
                # Add disc to destination
                env.state[to_rod].append(disc)
                visualizer.update_state(env.state)
                
                time.sleep(0.5)
    
    from threading import Thread
    demo_thread = Thread(target=demo)
    demo_thread.start()
    
    sys.exit(app.exec())
