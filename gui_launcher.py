"""
GUI Launcher for Tower of Hanoi RL Trainer.
Provides a modern PyQt6 interface with integrated visualization.
"""
import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDialog, QTextEdit, QListWidget, QListWidgetItem,
    QDialogButtonBox, QMessageBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QSpinBox, QFormLayout, QCheckBox, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QFont, QPalette, QColor


class TrainingWorker(QObject):
    """Worker object for running training in a separate thread."""
    finished = pyqtSignal(str)  # Emits completion message
    progress = pyqtSignal(int, float, float)  # episode, epsilon, success_rate
    
    def __init__(self, env, agent, visualizer, config):
        super().__init__()
        self.env = env
        self.agent = agent
        self.visualizer = visualizer
        self.config = config
        self.should_stop = False
    
    def run(self):
        """Run the training loop."""
        from util import flatten_state
        import numpy as np
        
        state_size = np.prod(self.env.observation_space.shape)
        
        for episode in range(1, self.config['episodes'] + 1):
            if self.should_stop or self.visualizer.should_stop:
                self.finished.emit("Training stopped by user")
                return
            
            state = self.env._reset()
            done = False
            steps = 0
            total_reward = 0
            
            # Update visualizer at start of episode
            self.visualizer.update_state(self.env.state)
            self.visualizer.update_info(episode=episode, step=0, reward=0, epsilon=self.agent.epsilon)
            
            while not done and steps < 1000:
                flat_state = flatten_state(state, self.config['num_discs'])
                flat_state = np.reshape(flat_state, [1, state_size])
                
                action = self.agent.act(flat_state)
                
                # Get move details for visualization
                from_rod, to_rod = self.env.decode_action(action)
                disc = self.env.state[from_rod][-1] if self.env.state[from_rod] else None
                
                # Execute action
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                flat_next_state = flatten_state(next_state, self.config['num_discs'])
                flat_next_state = np.reshape(flat_next_state, [1, state_size])
                
                self.agent.remember(flat_state, action, reward, flat_next_state, done)
                state = next_state
                steps += 1
                
                # Visualize the move
                if disc:
                    self.visualizer.animate_move(from_rod, to_rod, disc)
                    self.visualizer.update_state(self.env.state)
                    self.visualizer.update_info(episode=episode, step=steps, reward=total_reward, epsilon=self.agent.epsilon)
            
            # Emit progress signal
            self.progress.emit(episode, self.agent.epsilon, 0.0)
            
            # Train agent
            if len(self.agent.memory) > self.config.get('batch_size', 32):
                self.agent.replay()
        
        self.finished.emit(f"Training completed {self.config['episodes']} episodes!")
    
    def stop(self):
        """Stop the training."""
        self.should_stop = True


class MainLauncher(QMainWindow):
    """Main launcher window with integrated visualization."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tower of Hanoi - RL Trainer")
        self.setMinimumSize(800, 700)
        
        # Training worker and thread
        self.training_thread = None
        self.training_worker = None
        
        # Create central widget with stacked layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Stacked widget to switch between menu and visualization
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Create menu page
        self.menu_page = self.create_menu_page()
        self.stacked_widget.addWidget(self.menu_page)
        
        # Create visualization container (will be populated when needed)
        self.viz_container = QWidget()
        self.stacked_widget.addWidget(self.viz_container)
        
        # Show menu by default
        self.stacked_widget.setCurrentWidget(self.menu_page)
        
        # Apply styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
        """)
    
    def create_menu_page(self):
        """Create the main menu page."""
        menu_widget = QWidget()
        layout = QVBoxLayout(menu_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Title
        title = QLabel("🗼  Tower of Hanoi - RL Trainer  🗼")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Train an AI agent to solve Tower of Hanoi using Deep Q-Learning")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_font = QFont()
        subtitle_font.setPointSize(12)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: #666;")
        layout.addWidget(subtitle)
        
        layout.addSpacing(20)
        
        # Buttons container
        buttons_widget = QWidget()
        buttons_layout = QVBoxLayout(buttons_widget)
        buttons_layout.setSpacing(15)
        
        # Create menu buttons
        self.create_menu_button(buttons_layout, "🎬  Demo", 
                               "Watch the optimal solution with animation",
                               self.on_demo)
        
        self.create_menu_button(buttons_layout, "🏋️  Train Model", 
                               "Train a new AI model with visualization",
                               self.on_train)
        
        self.create_menu_button(buttons_layout, "🧪  Test Model", 
                               "Test a trained model with visualization",
                               self.on_test)
        
        self.create_menu_button(buttons_layout, "⚡  Quick Train", 
                               "Fast training (500 episodes, no visualization)",
                               self.on_quick_train)
        
        self.create_menu_button(buttons_layout, "📊  Compare Models", 
                               "Compare performance of multiple models",
                               self.on_compare_models)
        
        self.create_menu_button(buttons_layout, "📈  Learning Reports", 
                               "View training session history and learning rates",
                               self.on_learning_reports)
        
        self.create_menu_button(buttons_layout, "📚  Tutorial", 
                               "Learn how Tower of Hanoi works",
                               self.on_tutorial)
        
        layout.addWidget(buttons_widget)
        layout.addStretch()
        
        # Exit button
        exit_btn = QPushButton("❌  Exit")
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn)
        
        return menu_widget
    
    def create_menu_button(self, layout, text, description, callback):
        """Create a styled menu button with description."""
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)
        
        btn = QPushButton(text)
        btn.setMinimumHeight(50)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """)
        btn.clicked.connect(callback)
        container_layout.addWidget(btn)
        
        desc_label = QLabel(description)
        desc_label.setStyleSheet("""
            color: #666;
            font-size: 11px;
            padding-left: 12px;
        """)
        container_layout.addWidget(desc_label)
        
        layout.addWidget(container)
    
    def show_visualization_page(self, visualizer_widget):
        """Switch to visualization page and embed the visualizer."""
        # Clear previous visualization
        old_layout = self.viz_container.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        
        # Create new layout
        layout = QVBoxLayout(self.viz_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Add back to menu button at top
        back_btn = QPushButton("⬅️  Back to Menu")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        back_btn.clicked.connect(self.show_menu_page)
        layout.addWidget(back_btn)
        
        # Add visualizer widget
        layout.addWidget(visualizer_widget)
        
        # Switch to visualization page
        self.stacked_widget.setCurrentWidget(self.viz_container)
    
    def show_menu_page(self):
        """Switch back to menu page."""
        # Stop any running training
        if self.training_worker:
            self.training_worker.stop()
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.quit()
            self.training_thread.wait(1000)  # Wait up to 1 second
        
        self.stacked_widget.setCurrentWidget(self.menu_page)
    
    def on_demo(self):
        """Show demo visualization."""
        # Import and create visualizer in embedded mode
        from toh import TowerOfHanoiEnv
        from visualizer import TowerOfHanoiVisualizer
        import time
        from threading import Thread
        
        env = TowerOfHanoiEnv(num_discs=3)
        visualizer = TowerOfHanoiVisualizer(env, num_discs=3, standalone=False)
        
        # Show in main window
        self.show_visualization_page(visualizer)
        
        # Run demo
        def solve_hanoi(n, source, destination, auxiliary, moves):
            if n == 1:
                moves.append((source, destination))
                return
            solve_hanoi(n - 1, source, auxiliary, destination, moves)
            moves.append((source, destination))
            solve_hanoi(n - 1, auxiliary, destination, source, moves)
        
        moves = []
        solve_hanoi(3, 0, 2, 1, moves)
        
        state = [[3, 2, 1], [], []]
        visualizer.update_state(state)
        visualizer.update_info(episode=1, step=0, reward=0)
        
        def run_demo():
            time.sleep(1)
            for i, (from_rod, to_rod) in enumerate(moves):
                if visualizer.should_stop:
                    break
                disc = state[from_rod][-1]
                state[from_rod].pop()
                visualizer.animate_move(from_rod, to_rod, disc)
                state[to_rod].append(disc)
                visualizer.update_state(state)
                visualizer.update_info(step=i+1, reward=-(i+1))
                time.sleep(0.3)
        
        demo_thread = Thread(target=run_demo, daemon=True)
        demo_thread.start()
    
    def on_train(self):
        """Show training configuration dialog and start training in same window."""
        dialog = TrainingDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            
            # Import training components
            from toh import TowerOfHanoiEnv
            from dqn_agent import DQNAgent
            from visualizer import TowerOfHanoiVisualizer
            import numpy as np
            
            # Initialize
            env = TowerOfHanoiEnv(num_discs=config['num_discs'])
            state_size = np.prod(env.observation_space.shape)
            action_size = env.action_space.n
            agent = DQNAgent(state_size, action_size)
            
            # Create visualizer and show it
            visualizer = TowerOfHanoiVisualizer(env, num_discs=config['num_discs'], standalone=False)
            self.show_visualization_page(visualizer)
            
            # Create worker and thread
            self.training_worker = TrainingWorker(env, agent, visualizer, config)
            self.training_thread = QThread()
            self.training_worker.moveToThread(self.training_thread)
            
            # Connect signals
            self.training_thread.started.connect(self.training_worker.run)
            self.training_worker.progress.connect(self.on_training_progress)
            self.training_worker.finished.connect(self.on_training_finished)
            self.training_worker.finished.connect(self.training_thread.quit)
            
            # Store references for updates
            self.current_visualizer = visualizer
            
            # Start training
            self.training_thread.start()
    
    def on_training_progress(self, episode, epsilon, success_rate):
        """Handle training progress updates (runs on main thread)."""
        if hasattr(self, 'current_visualizer'):
            self.current_visualizer.update_info(episode=episode, epsilon=epsilon, success_rate=success_rate)
            QApplication.processEvents()
    
    def on_training_finished(self, message):
        """Handle training completion (runs on main thread)."""
        QMessageBox.information(self, "Training Complete", message)
    
    def on_test(self):
        """Show model selection dialog for testing."""
        dialog = ModelSelectionDialog(self, title="Test Model")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            model_path = dialog.get_selected_model()
            if model_path:
                from test_model import load_and_test_model
                load_and_test_model(model_path, num_discs=3, num_episodes=50, visualize=True)
    
    def on_quick_train(self):
        """Run quick training."""
        reply = QMessageBox.question(
            self, 
            "Quick Train", 
            "Start fast training with 500 episodes?\nVisualization will be hidden for faster training.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Same as on_train but with visualization hidden
            from toh import TowerOfHanoiEnv
            from dqn_agent import DQNAgent
            from visualizer import TowerOfHanoiVisualizer
            import numpy as np
            
            env = TowerOfHanoiEnv(num_discs=3)
            state_size = np.prod(env.observation_space.shape)
            action_size = env.action_space.n
            agent = DQNAgent(state_size, action_size)
            
            visualizer = TowerOfHanoiVisualizer(env, num_discs=3, standalone=False)
            visualizer.show_visualization = False  # Hide for speed
            self.show_visualization_page(visualizer)
            
            # Create worker and thread
            config = {'episodes': 500, 'num_discs': 3, 'batch_size': 32}
            self.training_worker = TrainingWorker(env, agent, visualizer, config)
            self.training_thread = QThread()
            self.training_worker.moveToThread(self.training_thread)
            
            # Connect signals
            self.training_thread.started.connect(self.training_worker.run)
            self.training_worker.progress.connect(self.on_training_progress)
            self.training_worker.finished.connect(self.on_training_finished)
            self.training_worker.finished.connect(self.training_thread.quit)
            
            # Store references for updates
            self.current_visualizer = visualizer
            
            # Start training
            self.training_thread.start()
    
    def on_compare_models(self):
        """Show model comparison dialog."""
        dialog = ModelComparisonDialog(self)
        dialog.exec()
    
    def on_learning_reports(self):
        """Show learning reports dialog."""
        dialog = LearningReportsDialog(self)
        dialog.exec()
    
    def on_tutorial(self):
        """Show tutorial dialog."""
        dialog = TutorialDialog(self)
        dialog.exec()


class TrainingDialog(QDialog):
    """Dialog for configuring training parameters."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Training")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Form layout for parameters
        form = QFormLayout()
        
        self.num_discs_spin = QSpinBox()
        self.num_discs_spin.setRange(3, 5)
        self.num_discs_spin.setValue(3)
        form.addRow("Number of Discs:", self.num_discs_spin)
        
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(100, 10000)
        self.episodes_spin.setSingleStep(100)
        self.episodes_spin.setValue(1000)
        form.addRow("Training Episodes:", self.episodes_spin)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 128)
        self.batch_spin.setSingleStep(16)
        self.batch_spin.setValue(32)
        form.addRow("Batch Size:", self.batch_spin)
        
        layout.addLayout(form)
        
        # Info label
        info = QLabel(
            "💡 Tip: More episodes = better learning but longer training time.\n"
            "Start with 1000 episodes for 3 discs."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(info)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_config(self):
        """Get training configuration."""
        return {
            'num_discs': self.num_discs_spin.value(),
            'episodes': self.episodes_spin.value(),
            'batch_size': self.batch_spin.value()
        }


class ModelSelectionDialog(QDialog):
    """Dialog for selecting a trained model."""
    
    def __init__(self, parent=None, title="Select Model"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        label = QLabel("Select a model to test:")
        layout.addWidget(label)
        
        # Model list
        self.model_list = QListWidget()
        self.load_models()
        layout.addWidget(self.model_list)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def load_models(self):
        """Load available models."""
        models_dir = Path("models")
        if not models_dir.exists():
            return
        
        model_files = sorted(models_dir.glob("*.weights.h5"), reverse=True)
        
        if not model_files:
            item = QListWidgetItem("No models found. Train a model first!")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.model_list.addItem(item)
            return
        
        for model_file in model_files:
            # Extract timestamp from filename
            try:
                parts = model_file.stem.split('_')
                if len(parts) >= 3:
                    date_str = parts[-2]
                    time_str = parts[-1]
                    timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                    display_text = f"{model_file.name}\n  Created: {timestamp}"
                else:
                    display_text = model_file.name
            except:
                display_text = model_file.name
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, str(model_file))
            self.model_list.addItem(item)
    
    def get_selected_model(self):
        """Get the selected model path."""
        current = self.model_list.currentItem()
        if current:
            return current.data(Qt.ItemDataRole.UserRole)
        return None


class ModelComparisonDialog(QDialog):
    """Dialog for comparing multiple models."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare Models")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        label = QLabel("Select 2 or more models to compare (Ctrl+Click for multiple):")
        layout.addWidget(label)
        
        # Model list (multi-select)
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.load_models()
        layout.addWidget(self.model_list)
        
        # Results area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.hide()
        layout.addWidget(self.results_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.compare_btn = QPushButton("📊 Compare Selected Models")
        self.compare_btn.clicked.connect(self.compare_models)
        button_layout.addWidget(self.compare_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def load_models(self):
        """Load available models."""
        models_dir = Path("models")
        if not models_dir.exists():
            return
        
        model_files = sorted(models_dir.glob("*.weights.h5"), reverse=True)
        
        if not model_files:
            item = QListWidgetItem("No models found. Train models first!")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.model_list.addItem(item)
            return
        
        for model_file in model_files:
            item = QListWidgetItem(model_file.name)
            item.setData(Qt.ItemDataRole.UserRole, str(model_file))
            self.model_list.addItem(item)
    
    def compare_models(self):
        """Compare selected models."""
        selected_items = self.model_list.selectedItems()
        
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Selection Error", "Please select at least 2 models to compare.")
            return
        
        model_paths = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        
        # Show progress
        self.compare_btn.setEnabled(False)
        self.compare_btn.setText("Comparing... Please wait")
        QApplication.processEvents()
        
        try:
            # Import and run comparison
            from test_model import compare_models
            import io
            import sys
            
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            compare_models(model_paths, num_discs=3, num_episodes=100)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Display results
            output = captured_output.getvalue()
            self.results_text.setPlainText(output)
            self.results_text.show()
            
            QMessageBox.information(self, "Comparison Complete", "Model comparison finished! See results below.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error comparing models: {str(e)}")
        
        finally:
            self.compare_btn.setEnabled(True)
            self.compare_btn.setText("📊 Compare Selected Models")


class LearningReportsDialog(QDialog):
    """Dialog for viewing learning reports."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Learning Reports")
        self.setMinimumSize(700, 500)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        label = QLabel("Training Session History:")
        layout.addWidget(label)
        
        # Reports text area
        self.reports_text = QTextEdit()
        self.reports_text.setReadOnly(True)
        layout.addWidget(self.reports_text)
        
        # Load reports
        self.load_reports()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
    
    def load_reports(self):
        """Load and display learning reports."""
        try:
            from model_evaluation import LearningRateTracker
            import io
            import sys
            
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            tracker = LearningRateTracker()
            tracker.compare_learning_rates()
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Display results
            output = captured_output.getvalue()
            if output.strip():
                self.reports_text.setPlainText(output)
            else:
                self.reports_text.setPlainText("No training reports found.\n\nTrain a model first to generate reports!")
        
        except Exception as e:
            self.reports_text.setPlainText(f"Error loading reports: {str(e)}")


class TutorialDialog(QDialog):
    """Dialog showing Tower of Hanoi tutorial."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tower of Hanoi Tutorial")
        self.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Tutorial content
        tutorial_text = QTextEdit()
        tutorial_text.setReadOnly(True)
        tutorial_text.setHtml("""
        <h2>🗼 Tower of Hanoi - How It Works</h2>
        
        <h3>📖 The Rules</h3>
        <p>The Tower of Hanoi is a classic puzzle with these rules:</p>
        <ol>
            <li><b>Setup:</b> You have 3 rods (A, B, C) and N discs of different sizes</li>
            <li><b>Starting Position:</b> All discs start on rod A, largest at bottom</li>
            <li><b>Goal:</b> Move all discs to rod C</li>
            <li><b>Movement Rules:</b>
                <ul>
                    <li>Only one disc can be moved at a time</li>
                    <li>A disc can only be placed on top of a larger disc (or empty rod)</li>
                    <li>A disc must be moved from the top of one rod to another</li>
                </ul>
            </li>
        </ol>
        
        <h3>🎯 Optimal Solution</h3>
        <p>The minimal solution requires <b>2<sup>N</sup> - 1</b> moves.</p>
        <ul>
            <li>For 3 discs: <b>7 moves minimum</b></li>
            <li>For 4 discs: 15 moves minimum</li>
            <li>For 5 discs: 31 moves minimum</li>
        </ul>
        
        <h3>🤖 How the AI Learns</h3>
        <p>This application uses <b>Deep Q-Learning (DQN)</b> to train an AI agent:</p>
        <ul>
            <li><b>Neural Network:</b> Learns to predict the best action for each state</li>
            <li><b>Exploration:</b> Tries random moves early to discover solutions</li>
            <li><b>Exploitation:</b> Uses learned knowledge to make optimal moves</li>
            <li><b>Experience Replay:</b> Remembers past experiences to learn efficiently</li>
            <li><b>Rewards:</b> +100 for solving, -1 per move, -10 for invalid moves</li>
        </ul>
        
        <h3>📊 Training Process</h3>
        <p>During training, the agent:</p>
        <ol>
            <li>Starts with random moves (high exploration)</li>
            <li>Gradually learns which moves lead to success</li>
            <li>Reduces exploration as it gains confidence</li>
            <li>Eventually discovers the optimal 7-move solution</li>
        </ol>
        
        <h3>💡 Tips for Best Results</h3>
        <ul>
            <li>Start with 1000 episodes for 3 discs</li>
            <li>Watch the visualization to see learning progress</li>
            <li>Use "Hide Visualization" for faster training</li>
            <li>Compare multiple models to find best settings</li>
            <li>A good model achieves 90%+ success rate with 80%+ efficiency</li>
        </ul>
        
        <h3>🎮 Try It Yourself!</h3>
        <p>Use the <b>Demo</b> button to see the optimal solution, then <b>Train</b> to watch 
        the AI learn from scratch!</p>
        """)
        layout.addWidget(tutorial_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


def main():
    """Launch the GUI."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainLauncher()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
