"""
GUI Launcher for Tower of Hanoi RL Trainer.
Provides a modern PyQt6 interface with integrated visualization.
"""
import sys
import os
import copy
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDialog, QTextEdit, QListWidget, QListWidgetItem,
    QDialogButtonBox, QMessageBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QSpinBox, QFormLayout, QCheckBox, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QMetaObject
from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6 import sip

# Matplotlib for performance graphs
import matplotlib
matplotlib.use('QtAgg')  # Use Qt backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class TrainingWorker(QObject):
    """Worker object for running training in a separate thread."""
    finished = pyqtSignal(str)  # Emits completion message
    progress = pyqtSignal(int, float, float)  # episode, epsilon, success_rate
    model_saved = pyqtSignal(str, object)  # model_path, metadata
    update_info = pyqtSignal(dict)  # GUI update data (episode, step, reward, epsilon)
    update_state = pyqtSignal(object)  # Environment state update
    animate_move = pyqtSignal(int, int, int)  # from_rod, to_rod, disc
    performance_data = pyqtSignal(int, float, float)  # episode, steps, moving_avg_steps
    
    def __init__(self, env, agent, visualizer, config):
        super().__init__()
        self.env = env
        self.agent = agent
        self.visualizer = visualizer
        self.config = config
        self.should_stop = False
        self.success_count = 0
        self.total_steps = 0
        
        # Early stopping parameters
        self.patience = config.get('patience', 50)  # Stop if no improvement for N episodes
        self.min_improvement = 0.05  # Minimum 5% improvement required
        self.best_avg_steps = float('inf')
        self.episodes_without_improvement = 0
    
    def run(self):
        """Run the training loop."""
        from util import flatten_state
        import numpy as np
        from model_manager import ModelManager
        
        # Use agent's state_size (which supports maximum discs) instead of env's
        state_size = self.agent.state_size
        self.success_count = 0
        self.total_steps = 0
        episode_steps_list = []
        
        # Send initial configuration to visualizer
        self.update_info.emit({
            'target_episodes': self.config['episodes'],
            'episode': 0,
            'step': 0,
            'reward': 0,
            'epsilon': self.agent.epsilon,
            'success_rate': 0.0,
            'avg_steps': 0.0
        })
        
        for episode in range(1, self.config['episodes'] + 1):
            if self.should_stop or self.visualizer.should_stop:
                # Save model before stopping
                self._save_current_model(episode - 1, episode_steps_list)
                self.finished.emit(f"Training stopped at episode {episode-1}. Model saved.")
                return
            
            # Reset episode-specific agent state
            self.agent.reset_episode()
            
            state = self.env._reset()
            done = False
            steps = 0
            total_reward = 0
            
            # Update visualizer at start of episode
            self.update_state.emit(copy.deepcopy(self.env.state))
            self.update_info.emit({'episode': episode, 'step': 0, 'reward': 0, 'epsilon': self.agent.epsilon})
            
            while not done and steps < 1000:
                flat_state = flatten_state(state, self.config['num_discs'])
                
                # Pad state to match agent's state_size (for generalization across disc counts)
                if len(flat_state) < state_size:
                    flat_state = np.pad(flat_state, (0, state_size - len(flat_state)), 'constant')
                
                flat_state = np.reshape(flat_state, [1, state_size])
                
                # Get valid actions to avoid learning from invalid moves
                valid_actions = self.env.get_valid_actions()
                action = self.agent.act(flat_state, valid_actions)
                
                # Get move details for visualization BEFORE executing step
                from_rod, to_rod = self.env.decode_action(action)
                disc = self.env.state[from_rod][-1] if self.env.state[from_rod] else None
                
                # Execute action (this modifies env.state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                # Log significant rewards/penalties for debugging
                if abs(reward) > 5:  # Log notable rewards
                    reward_type = "REWARD" if reward > 0 else "PENALTY"
                    print(f"Episode {episode}, Step {steps}: {reward_type} = {reward:.1f} (action: {from_rod}→{to_rod}, disc: {disc})")
                
                flat_next_state = flatten_state(next_state, self.config['num_discs'])
                
                # Pad next state to match agent's state_size
                if len(flat_next_state) < state_size:
                    flat_next_state = np.pad(flat_next_state, (0, state_size - len(flat_next_state)), 'constant')
                
                flat_next_state = np.reshape(flat_next_state, [1, state_size])
                
                self.agent.remember(flat_state, action, reward, flat_next_state, done)
                state = next_state
                steps += 1
                
                # Train agent after each step if enough memory
                if len(self.agent.memory) > self.config.get('batch_size', 64):
                    self.agent.replay()
                
                # Visualize the move
                if disc:
                    self.animate_move.emit(from_rod, to_rod, disc)
                    self.update_state.emit(copy.deepcopy(self.env.state))
                    self.update_info.emit({'episode': episode, 'step': steps, 'reward': total_reward, 'epsilon': self.agent.epsilon})
                    
                    # If visualization is enabled, add a small delay to allow animation to render
                    # This is checked via the visualizer's show_visualization flag
                    if self.visualizer.show_visualization:
                        import time
                        # Wait for animation to complete (animation_speed * animation_frames)
                        time.sleep((self.visualizer.animation_speed * self.visualizer.animation_frames) / 1000.0)
            
            # Track success
            success = (len(self.env.state[2]) == self.config['num_discs'])
            if success:
                self.success_count += 1
            episode_steps_list.append(steps)
            self.total_steps += steps
            
            # Calculate metrics
            success_rate = (self.success_count / episode) * 100
            avg_steps = self.total_steps / episode
            
            # Calculate moving average for last 20 episodes for early stopping
            window_size = min(20, episode)
            recent_steps = episode_steps_list[-window_size:]
            moving_avg_steps = sum(recent_steps) / len(recent_steps)
            
            # Emit performance data for graphing
            self.performance_data.emit(episode, steps, moving_avg_steps)
            
            # Early stopping check (after initial exploration period)
            if episode >= 100:  # Allow initial learning phase
                # Check if performance is improving
                if moving_avg_steps < self.best_avg_steps * (1 - self.min_improvement):
                    # Significant improvement!
                    self.best_avg_steps = moving_avg_steps
                    self.episodes_without_improvement = 0
                    print(f"Episode {episode}: New best avg steps: {moving_avg_steps:.1f}")
                else:
                    self.episodes_without_improvement += 1
                    
                    # Check if we should stop
                    if self.episodes_without_improvement >= self.patience:
                        print(f"\nEarly stopping triggered at episode {episode}")
                        print(f"No improvement for {self.patience} episodes")
                        print(f"Best moving average: {self.best_avg_steps:.1f} steps")
                        self._save_current_model(episode, episode_steps_list)
                        self.finished.emit(f"Training stopped early at episode {episode} (no improvement). Model saved.")
                        return
            
            # Emit progress signal with comprehensive metrics
            self.progress.emit(episode, self.agent.epsilon, success_rate)
            
            # Send detailed update to visualizer
            self.update_info.emit({
                'episode': episode,
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'episode_success': 1 if success else 0,
                'epsilon': self.agent.epsilon,
                'target_episodes': self.config['episodes']
            })
            
            # Save checkpoint every 100 episodes (to prevent data loss)
            if episode % 100 == 0 and episode < self.config['episodes']:
                self._save_current_model(episode, episode_steps_list)
                print(f"Checkpoint saved at episode {episode}")
            
            # Note: Agent training now happens after each step, not per episode
        
        # Save model after training completes normally
        self._save_current_model(self.config['episodes'], episode_steps_list)
        self.finished.emit(f"Training completed {self.config['episodes']} episodes! Model saved.")
    
    def _save_current_model(self, episodes_completed, episode_steps_list):
        """Save the current model with metadata."""
        from model_manager import ModelManager
        
        model_manager = ModelManager()
        
        # Calculate metrics
        if episodes_completed > 0:
            avg_steps = self.total_steps / episodes_completed
            success_rate = (self.success_count / episodes_completed) * 100
        else:
            avg_steps = 0
            success_rate = 0
        
        metadata = {
            'episodes': episodes_completed,
            'num_discs': self.config['num_discs'],
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'final_epsilon': float(self.agent.epsilon),
            'total_steps': self.total_steps
        }
        
        model_dir = model_manager.save_model(self.agent, metadata=metadata)
        self.model_saved.emit(str(model_dir), metadata)
    
    def stop(self):
        """Stop the training."""
        self.should_stop = True


class PerformanceGraphWidget(QWidget):
    """Widget to display real-time performance graph during training."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.episodes = []
        self.steps_data = []
        self.moving_avg_data = []
        self.optimal_steps = 7  # For 3 discs, will be updated dynamically
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Setup layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Initialize plot
        self.init_plot()
    
    def init_plot(self):
        """Initialize the plot with labels and styling."""
        self.ax.clear()
        self.ax.set_xlabel('Episode', fontsize=10)
        self.ax.set_ylabel('Steps', fontsize=10)
        self.ax.set_title('Training Performance: Steps per Episode', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#f0f0f0')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def set_optimal_steps(self, num_discs):
        """Set the optimal steps line based on number of discs."""
        self.optimal_steps = (2 ** num_discs) - 1
    
    def update_plot(self, episode, steps, moving_avg):
        """Update the plot with new data point."""
        self.episodes.append(episode)
        self.steps_data.append(steps)
        self.moving_avg_data.append(moving_avg)
        
        # Keep only last 500 episodes for performance
        if len(self.episodes) > 500:
            self.episodes = self.episodes[-500:]
            self.steps_data = self.steps_data[-500:]
            self.moving_avg_data = self.moving_avg_data[-500:]
        
        # Redraw plot
        self.ax.clear()
        
        # Plot steps per episode (light blue, semi-transparent)
        self.ax.plot(self.episodes, self.steps_data, 'o-', color='lightblue', 
                     linewidth=1, markersize=2, alpha=0.5, label='Steps per episode')
        
        # Plot moving average (thick blue line)
        self.ax.plot(self.episodes, self.moving_avg_data, '-', color='#2196F3', 
                     linewidth=2.5, label='Moving average (20 episodes)')
        
        # Plot optimal steps line (green dashed)
        if len(self.episodes) > 0:
            self.ax.axhline(y=self.optimal_steps, color='green', linestyle='--', 
                           linewidth=2, label=f'Optimal ({self.optimal_steps} steps)')
        
        # Styling
        self.ax.set_xlabel('Episode', fontsize=10)
        self.ax.set_ylabel('Steps', fontsize=10)
        self.ax.set_title('Training Performance: Steps per Episode', fontsize=12, fontweight='bold')
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#f0f0f0')
        
        # Set y-axis limits for better visualization
        if len(self.steps_data) > 0:
            max_steps = max(self.steps_data[-min(100, len(self.steps_data)):])
            self.ax.set_ylim(0, max(max_steps * 1.1, self.optimal_steps * 2))
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def clear_plot(self):
        """Clear all data and reset the plot."""
        self.episodes = []
        self.steps_data = []
        self.moving_avg_data = []
        self.init_plot()


class MainLauncher(QMainWindow):
    """Main launcher window with integrated visualization."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tower of Hanoi - RL Trainer")
        self.setMinimumSize(1400, 900)  # Updated to match visualizer window size
        
        # Training worker and thread
        self.training_thread = None
        self.training_worker = None
        self.current_agent = None  # Store current agent for architecture display
        
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
        layout.setSpacing(10)
        layout.setContentsMargins(30, 15, 30, 15)  # Further reduced margins
        
        # Title
        title = QLabel("🗼  Tower of Hanoi - RL Trainer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)  # Reduced from 20
        title_font.setBold(True)
        title.setFont(title_font)
        title.setMaximumHeight(40)  # Prevent title from expanding
        layout.addWidget(title, 0)  # 0 = no stretch
        
        # Small spacing after title
        layout.addSpacing(5)
        
        # Buttons container
        buttons_widget = QWidget()
        buttons_widget.setMaximumHeight(560)  # 8 buttons * 62px + 7 gaps * 8px = 552px
        buttons_layout = QVBoxLayout(buttons_widget)
        buttons_layout.setSpacing(8)  # Reduced from 15 to make more compact
        
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
        
        self.create_menu_button(buttons_layout, "�  Continue Training", 
                               "Load existing model and continue training (transfer learning)",
                               self.on_continue_training)
        
        self.create_menu_button(buttons_layout, "�📊  Compare Models", 
                               "Compare performance of multiple models",
                               self.on_compare_models)
        
        self.create_menu_button(buttons_layout, "📈  Learning Reports", 
                               "View training session history and learning rates",
                               self.on_learning_reports)
        
        self.create_menu_button(buttons_layout, "📚  Tutorial", 
                               "Learn how Tower of Hanoi works",
                               self.on_tutorial)
        
        layout.addWidget(buttons_widget, 0)  # 0 = no stretch, keep compact
        
        # Exit button
        exit_btn = QPushButton("❌  Exit")
        exit_btn.setMinimumHeight(38)
        exit_btn.setMaximumHeight(38)  # Prevent expansion
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 8px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn, 0)  # 0 = no stretch
        
        return menu_widget
    
    def create_menu_button(self, layout, text, description, callback):
        """Create a styled menu button with description."""
        container = QWidget()
        container.setMaximumHeight(62)  # Fix height: 42 (button) + 15 (desc) + 5 (spacing)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)
        
        btn = QPushButton(text)
        btn.setMinimumHeight(42)  # Reduced from 50
        btn.setMaximumHeight(42)  # Prevent expansion
        btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 12px;
                font-size: 15px;
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
            font-size: 10px;
            padding-left: 12px;
        """)
        container_layout.addWidget(desc_label)
        
        layout.addWidget(container)
    
    def show_visualization_page(self, visualizer_widget, show_test_again=False, performance_graph=None):
        """Switch to visualization page and embed the visualizer."""
        # Clear previous visualization
        old_layout = self.viz_container.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item.widget():
                    widget = item.widget()
                    if widget and widget != visualizer_widget and widget != performance_graph:  # Don't delete new widgets
                        widget.deleteLater()
        
        # Set the new visualizer as current BEFORE clearing reference
        self.current_visualizer = visualizer_widget
        
        # Create new layout
        layout = QVBoxLayout(self.viz_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Top button bar
        button_bar = QHBoxLayout()
        
        # Add back to menu button
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
        button_bar.addWidget(back_btn)
        
        # Add Test Again button if in test mode
        if show_test_again:
            button_bar.addStretch()
            test_again_btn = QPushButton("🔄 Test Again")
            test_again_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """)
            test_again_btn.clicked.connect(self.retest_model)
            button_bar.addWidget(test_again_btn)
        
        button_bar_widget = QWidget()
        button_bar_widget.setLayout(button_bar)
        layout.addWidget(button_bar_widget)
        
        # If performance graph is provided, add it at the bottom (for training mode)
        if performance_graph:
            # Create vertical layout: visualizer on top, graph on bottom
            content_layout = QVBoxLayout()
            content_layout.addWidget(visualizer_widget, stretch=2)  # Visualizer takes more space
            content_layout.addWidget(performance_graph, stretch=1)  # Graph at bottom
            content_widget = QWidget()
            content_widget.setLayout(content_layout)
            layout.addWidget(content_widget)
        else:
            # Just add visualizer (for test mode)
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
        visualizer.update_info({'episode': 1, 'step': 0, 'reward': 0})
        
        # Create a demo worker that emits signals
        class DemoWorker(QObject):
            animate_move_signal = pyqtSignal(int, int, int)
            update_state_signal = pyqtSignal(object)
            update_info_signal = pyqtSignal(dict)
            finished = pyqtSignal()
            
            def __init__(self, moves, state, visualizer):
                super().__init__()
                self.moves = moves
                self.state = [list(rod) for rod in state]  # Deep copy
                self.visualizer = visualizer
            
            def run(self):
                import time
                time.sleep(1)
                for i, (from_rod, to_rod) in enumerate(self.moves):
                    if self.visualizer.should_stop:
                        break
                    disc = self.state[from_rod][-1]
                    self.state[from_rod].pop()
                    self.animate_move_signal.emit(from_rod, to_rod, disc)
                    self.state[to_rod].append(disc)
                    time.sleep(0.3)
                    self.update_state_signal.emit([list(rod) for rod in self.state])
                    self.update_info_signal.emit({'step': i+1, 'reward': -(i+1)})
                    time.sleep(0.3)
                self.finished.emit()
        
        worker = DemoWorker(moves, state, visualizer)
        thread = QThread()
        worker.moveToThread(thread)
        
        # Connect signals
        thread.started.connect(worker.run)
        worker.animate_move_signal.connect(visualizer.animate_move)
        worker.update_state_signal.connect(visualizer.update_state)
        worker.update_info_signal.connect(visualizer.update_info)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        
        # Store references to prevent garbage collection
        self.demo_worker = worker
        self.demo_thread = thread
        
        thread.start()
    
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
            
            # Use MAXIMUM state size to support multiple disc counts
            # This allows the model to generalize across different disc counts
            MAX_DISCS = 10  # Maximum discs we want to support
            state_size = MAX_DISCS * 3  # 3 rods, up to 10 discs each
            action_size = env.action_space.n
            
            # Show info about generalization capability
            QMessageBox.information(
                self,
                "Training Configuration",
                f"Training with {config['num_discs']} discs.\n\n"
                f"Model capacity: Up to {MAX_DISCS} discs\n"
                f"State size: {state_size} (padded for generalization)\n\n"
                f"This allows the model to work with other disc counts\n"
                f"(3-{MAX_DISCS}) if trained on multiple configurations."
            )
            
            # Create agent with selected architecture
            agent = DQNAgent(state_size, action_size, architecture_name=config['architecture'])
            
            # Create visualizer
            visualizer = TowerOfHanoiVisualizer(env, num_discs=config['num_discs'], standalone=False)
            
            # Create performance graph
            self.performance_graph = PerformanceGraphWidget()
            self.performance_graph.set_optimal_steps(config['num_discs'])
            
            # Show visualization page with graph
            self.show_visualization_page(visualizer, performance_graph=self.performance_graph)
            
            # Store agent for later access
            self.current_agent = agent
            
            # Create worker and thread
            self.training_worker = TrainingWorker(env, agent, visualizer, config)
            self.training_thread = QThread()
            self.training_worker.moveToThread(self.training_thread)
            
            # Connect signals
            self.training_thread.started.connect(self.training_worker.run)
            self.training_worker.progress.connect(self.on_training_progress)
            self.training_worker.finished.connect(self.on_training_finished)
            self.training_worker.model_saved.connect(self.on_model_saved)
            self.training_worker.update_info.connect(visualizer.update_info)
            self.training_worker.update_state.connect(visualizer.update_state)
            self.training_worker.animate_move.connect(visualizer.animate_move)
            self.training_worker.performance_data.connect(self.on_performance_update)
            self.training_worker.finished.connect(self.training_thread.quit)
            
            # Store references for updates
            self.current_visualizer = visualizer
            
            # Start training
            self.training_thread.start()
    
    def on_training_progress(self, episode, epsilon, success_rate):
        """Handle training progress updates (runs on main thread)."""
        if hasattr(self, 'current_visualizer') and self.current_visualizer is not None:
            try:
                # Check if the visualizer widget still exists
                if not sip.isdeleted(self.current_visualizer):
                    self.current_visualizer.update_info({'episode': episode, 'epsilon': epsilon, 'success_rate': success_rate})
                    QApplication.processEvents()
            except RuntimeError:
                # Widget was deleted, clear reference
                self.current_visualizer = None
    
    def on_performance_update(self, episode, steps, moving_avg):
        """Handle performance data updates for the graph."""
        if hasattr(self, 'performance_graph') and self.performance_graph is not None:
            try:
                if not sip.isdeleted(self.performance_graph):
                    self.performance_graph.update_plot(episode, steps, moving_avg)
                    QApplication.processEvents()
            except RuntimeError:
                # Widget was deleted, clear reference
                self.performance_graph = None
    
    def on_training_finished(self, message):
        """Handle training completion (runs on main thread)."""
        QMessageBox.information(self, "Training Complete", message)
    
    def on_model_saved(self, model_path, metadata):
        """Handle model save completion - show architecture"""
        if self.current_agent and self.current_agent.model:
            reply = QMessageBox.question(
                self,
                "Model Saved",
                f"Model saved successfully!\n\nSuccess Rate: {metadata.get('success_rate', 0):.1f}%\n"
                f"Avg Steps: {metadata.get('avg_steps', 0):.1f}\n\n"
                "Would you like to view the model architecture?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.show_model_architecture(self.current_agent.model, metadata)
    
    def on_test(self):
        """Show model selection dialog for testing."""
        from model_selection_dialog import ModelSelectionDialog
        from model_manager import ModelManager
        
        try:
            dialog = ModelSelectionDialog(self, auto_select_latest=True)
            result = dialog.exec()
            
            if result == QDialog.DialogCode.Accepted:
                model_name, metadata = dialog.get_selected_model()
                if not model_name:
                    QMessageBox.warning(self, "No Selection", "Please select a model to test.")
                    return
                
                # Load the model
                model_manager = ModelManager()
                try:
                    agent, metadata = model_manager.load_model(model_name)
                    
                    # Show model architecture first
                    self.show_model_architecture(agent.model, metadata)
                    
                    # Calculate maximum disc count supported by model
                    max_discs = agent.state_size // 3
                    
                    # Show test configuration dialog with metadata
                    test_config = self.show_test_config_dialog(metadata.get('num_discs', 3), agent.state_size, max_discs, metadata)
                    if not test_config:
                        return  # User cancelled
                    
                    num_discs = test_config['num_discs']
                    show_visualization = test_config['show_visualization']
                    exploration = test_config['exploration']
                    
                    # Validate disc count is within model capacity
                    if num_discs > max_discs:
                        QMessageBox.warning(
                            self,
                            "Disc Count Too High",
                            f"⚠️ This model's state size ({agent.state_size}) can only handle up to {max_discs} discs.\n\n"
                            f"Testing with {num_discs} discs is not possible.\n\n"
                            f"Please select {max_discs} discs or fewer.",
                            QMessageBox.StandardButton.Ok
                        )
                        return
                    
                    # Run test with visualization
                    from toh import TowerOfHanoiEnv
                    from visualizer import TowerOfHanoiVisualizer
                    from util import flatten_state
                    import numpy as np
                    
                    env = TowerOfHanoiEnv(num_discs=num_discs)
                    visualizer = TowerOfHanoiVisualizer(env, num_discs=num_discs, standalone=False)
                    
                    # Store test context for retesting
                    self.test_env = env
                    self.test_agent = agent
                    self.test_num_discs = num_discs
                    self.test_show_visualization = show_visualization
                    self.test_exploration = exploration
                    self.test_metadata = metadata
                    self.test_trained_num_discs = agent.state_size // 3  # Store original training disc count
                    
                    self.show_visualization_page(visualizer, show_test_again=True)
                    
                    # Run test episode with exploration
                    self.run_test_episode(env, agent, visualizer, num_discs, show_visualization, exploration, metadata)
                        
                except Exception as e:
                    import traceback
                    error_msg = f"Failed to load model:\n{str(e)}\n\n{traceback.format_exc()}"
                    QMessageBox.critical(self, "Error", error_msg)
        except Exception as e:
            import traceback
            error_msg = f"Error in test dialog:\n{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
    
    def show_test_config_dialog(self, default_num_discs, model_state_size, max_discs, metadata=None):
        """Show dialog to configure test parameters."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Test Configuration")
        dialog.setMinimumSize(450, 450)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title = QLabel("Configure Test Parameters")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Model info
        info = QLabel(f"Model state size: {model_state_size} (supports up to {max_discs} discs)")
        info.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)
        
        # Important notice about generalization
        notice = QLabel(
            f"💡 This model can test with {max_discs} discs or fewer.\n"
            f"If trained on multiple disc counts (e.g., 3 and 4), it can generalize to others.\n"
            f"States are automatically padded to match the model's capacity."
        )
        notice.setStyleSheet("color: #004085; background-color: #cce5ff; padding: 10px; border-radius: 5px; border: 1px solid #b8daff;")
        notice.setWordWrap(True)
        notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(notice)
        
        # Form layout
        form = QFormLayout()
        
        # Number of discs (user can select within capacity)
        discs_spin = QSpinBox()
        discs_spin.setMinimum(3)
        discs_spin.setMaximum(max_discs)
        discs_spin.setValue(min(default_num_discs, max_discs))
        discs_spin.setToolTip(f"Select number of discs for testing (3-{max_discs})")
        form.addRow("Number of Discs:", discs_spin)
        
        # Visualization toggle
        viz_check = QCheckBox("Show Visualization")
        viz_check.setChecked(True)
        viz_check.setToolTip("Uncheck for faster testing without animation")
        form.addRow("Visualization:", viz_check)
        
        # Exploration parameter (temperature for action selection)
        from PyQt6.QtWidgets import QSlider
        exploration_slider = QSlider(Qt.Orientation.Horizontal)
        exploration_slider.setMinimum(0)
        exploration_slider.setMaximum(100)
        exploration_slider.setValue(10)  # Default 10% exploration
        exploration_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        exploration_slider.setTickInterval(10)
        
        exploration_label = QLabel("10%")
        exploration_label.setStyleSheet("font-weight: bold; color: #007bff;")
        
        def update_exploration_label(value):
            exploration_label.setText(f"{value}%")
        
        exploration_slider.valueChanged.connect(update_exploration_label)
        
        exploration_layout = QHBoxLayout()
        exploration_layout.addWidget(exploration_slider, stretch=4)
        exploration_layout.addWidget(exploration_label, stretch=1)
        
        exploration_widget = QWidget()
        exploration_widget.setLayout(exploration_layout)
        
        form.addRow("Exploration:", exploration_widget)
        
        # Exploration help text
        exploration_help = QLabel(
            "💡 Exploration adds randomness to action selection.\n"
            "Higher values help escape loops but may make solving slower.\n"
            "Recommended: 10-20% for stuck models, 0% for well-trained models."
        )
        exploration_help.setStyleSheet("color: #666; font-size: 10px; padding: 5px; background-color: #f8f9fa; border-radius: 3px;")
        exploration_help.setWordWrap(True)
        form.addRow("", exploration_help)
        
        layout.addLayout(form)
        
        # Info about disc counts - update dynamically based on selection
        def update_info_text():
            selected_discs = discs_spin.value()
            if selected_discs <= 3:
                info_text = f"💡 {selected_discs} discs: Should solve quickly (optimal: {2**selected_discs - 1} moves)"
            elif selected_discs == 4:
                info_text = f"💡 {selected_discs} discs: May take 15-31 moves (optimal: 15 moves)"
            else:
                info_text = f"💡 {selected_discs} discs: May take longer (optimal: {2**selected_discs - 1} moves)"
            
            info_label.setText(info_text)
        
        info_label = QLabel()
        info_label.setStyleSheet("color: #004085; background-color: #cce5ff; padding: 10px; border-radius: 5px; border: 1px solid #b8daff;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Connect spinner to update info
        discs_spin.valueChanged.connect(update_info_text)
        update_info_text()  # Initialize
        
        layout.addStretch()
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return {
                'num_discs': discs_spin.value(),
                'show_visualization': viz_check.isChecked(),
                'exploration': exploration_slider.value() / 100.0,  # Convert to 0.0-1.0
                'metadata': metadata  # Pass along for avg_steps calculation
            }
        return None
    
    def show_model_architecture(self, model, metadata):
        """Show model architecture in a dialog"""
        from model_visualizer import ModelVisualizerWidget
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Model Architecture")
        dialog.setMinimumSize(900, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Model info
        info_text = f"<b>Model:</b> {metadata.get('name', 'Unknown')}<br>"
        info_text += f"<b>Episodes Trained:</b> {metadata.get('episodes', '-')}<br>"
        info_text += f"<b>Success Rate:</b> {metadata.get('success_rate', '-'):.1f}%<br>"
        info_text += f"<b>Avg Steps:</b> {metadata.get('avg_steps', '-'):.1f}"
        
        info_label = QLabel(info_text)
        info_label.setFont(QFont("Arial", 11))
        layout.addWidget(info_label)
        
        # Model visualizer
        viz = ModelVisualizerWidget()
        model_info = f"Episodes: {metadata.get('episodes', '-')} | Success: {metadata.get('success_rate', 0):.1f}%"
        viz.set_model(model, model_info)
        layout.addWidget(viz)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def run_test_episode(self, env, agent, visualizer, num_discs, show_visualization=True, exploration=0.1, metadata=None):
        """Run a single test episode with optional visualization and exploration"""
        from util import flatten_state
        import numpy as np
        
        class TestWorker(QObject):
            animate_move_signal = pyqtSignal(int, int, int)
            update_state_signal = pyqtSignal(object)
            update_info_signal = pyqtSignal(dict)
            show_message = pyqtSignal(str, str, bool)  # title, message, success
            finished = pyqtSignal()
            
            def __init__(self, env, agent, num_discs, show_visualization, exploration, metadata):
                super().__init__()
                self.env = env
                self.agent = agent
                self.num_discs = num_discs
                self.show_visualization = show_visualization
                self.exploration = exploration
                self.metadata = metadata
            
            def run(self):
                import time
                state = self.env._reset()
                self.update_state_signal.emit(self.env.state)
                self.update_info_signal.emit({'episode': 1, 'step': 0, 'reward': 0})
                time.sleep(1)
                
                done = False
                steps = 0
                total_reward = 0
                invalid_move_count = 0
                last_actions = []
                
                # Set reasonable step limit based on training performance
                # If we have avg_steps from training, use 2x that, otherwise use 10x optimal
                optimal_steps = (2 ** self.num_discs) - 1
                
                if self.metadata and 'avg_steps' in self.metadata:
                    # Use 2x average training steps, with a minimum of 5x optimal
                    avg_training_steps = self.metadata['avg_steps']
                    max_steps = max(int(avg_training_steps * 2), optimal_steps * 5)
                    print(f"Using training avg ({avg_training_steps:.1f}) -> max_steps: {max_steps}")
                else:
                    # No training data, use 10x optimal
                    max_steps = optimal_steps * 10
                
                print("\n=== TEST EPISODE STARTED ===")
                print(f"Initial state: {self.env.state}")
                print(f"Agent state_size: {self.agent.state_size}, Test num_discs: {self.num_discs}")
                print(f"Optimal steps: {optimal_steps}, Max steps: {max_steps}")
                print(f"Exploration rate: {self.exploration * 100:.0f}%")
                
                while not done and steps < max_steps:
                    flat_state = flatten_state(state, self.num_discs)
                    
                    # Handle state size mismatch by padding or truncating
                    if len(flat_state) < self.agent.state_size:
                        # Pad with zeros if state is smaller
                        flat_state = np.pad(flat_state, (0, self.agent.state_size - len(flat_state)), 'constant')
                        print(f"  Padded state from {len(flatten_state(state, self.num_discs))} to {self.agent.state_size}")
                    elif len(flat_state) > self.agent.state_size:
                        # Truncate if state is larger (not recommended)
                        flat_state = flat_state[:self.agent.state_size]
                        print(f"  ⚠️ Truncated state from {len(flatten_state(state, self.num_discs))} to {self.agent.state_size}")
                    
                    flat_state = np.reshape(flat_state, [1, self.agent.state_size])
                    
                    # Get Q-values for all actions
                    q_values = self.agent.model.predict(flat_state, verbose=0)
                    
                    # Mask invalid actions by setting their Q-values to -inf
                    valid_actions = self.env.get_valid_actions()
                    masked_q_values = q_values[0].copy()
                    for action in range(len(masked_q_values)):
                        if action not in valid_actions:
                            masked_q_values[action] = -np.inf
                    
                    # Epsilon-greedy action selection with configurable exploration
                    use_random = False
                    if np.random.random() < self.exploration:
                        # Explore: choose random valid action
                        action = np.random.choice(valid_actions)
                        use_random = True
                        print(f"  🎲 Exploration: Random action selected")
                    else:
                        # Exploit: select best action based on Q-values
                        action = np.argmax(masked_q_values)
                    
                    # Additional loop-breaking logic (less aggressive now with exploration)
                    if not use_random and len(last_actions) >= 6:
                        recent_6 = last_actions[-6:]
                        # If stuck in obvious pattern, force exploration
                        if len(set(recent_6)) <= 2:
                            action = np.random.choice(valid_actions)
                            use_random = True
                            print(f"  🔄 Loop detected: Forced random action")
                    
                    # Track repeating actions
                    last_actions.append(action)
                    if len(last_actions) > 10:
                        last_actions.pop(0)
                    
                    # Get move details BEFORE executing
                    from_rod, to_rod = self.env.decode_action(action)
                    disc = self.env.state[from_rod][-1] if self.env.state[from_rod] else None
                    
                    print(f"\nStep {steps + 1}:")
                    print(f"  State: {self.env.state}")
                    print(f"  Valid actions: {valid_actions}")
                    print(f"  Action: {action} ({from_rod}→{to_rod})")
                    print(f"  Q-values (raw): {q_values[0]}")
                    print(f"  Q-values (masked): {masked_q_values}")
                    print(f"  Top disc on rod {from_rod}: {disc}")
                    
                    # Execute action
                    next_state, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    print(f"  Reward: {reward}, Total: {total_reward}")
                    print(f"  Valid move: {disc is not None}")
                    
                    # Track TRULY invalid moves (reward <= -40 indicates serious rule violation)
                    # The -50 penalty from environment might be too strict with reward shaping
                    if reward <= -40:
                        invalid_move_count += 1
                        print(f"  ⚠️ TRULY INVALID MOVE (rule violation)! Count: {invalid_move_count}")
                        
                        # More lenient: allow up to 5 invalid moves before stopping
                        if invalid_move_count > 5:
                            print(f"  ⚠️ TOO MANY INVALID MOVES!")
                            print(f"  Last 10 actions: {last_actions}")
                            self.show_message.emit("Test Stopped", 
                                                  f"Agent made {invalid_move_count} rule-violating moves in {steps} steps.",
                                                  False)
                            break
                    elif reward < 0:
                        # Negative reward but not invalid move (oscillation, inefficiency, etc.)
                        print(f"  ℹ️ Suboptimal move (penalty: {reward})")
                    
                    # Detect oscillation/stuck patterns regardless of reward
                    if len(last_actions) >= 6:
                        # Check if stuck in short loop (repeating 2-3 actions)
                        recent_6 = last_actions[-6:]
                        if (recent_6[0] == recent_6[2] == recent_6[4] and 
                            recent_6[1] == recent_6[3] == recent_6[5]):
                            print("  ⚠️ STUCK IN 2-ACTION LOOP - STOPPING TEST")
                            print(f"  Pattern: {recent_6}")
                            self.show_message.emit("Test Stopped", 
                                                  f"Agent stuck in oscillation loop after {steps} steps.",
                                                  False)
                            break
                        
                        # Check if last 8 moves are just 2-3 unique actions
                        if len(last_actions) >= 8:
                            recent_8 = last_actions[-8:]
                            unique_actions = len(set(recent_8))
                            if unique_actions <= 2:
                                print(f"  ⚠️ STUCK - Only {unique_actions} unique actions in last 8 moves")
                                print(f"  Last actions: {last_actions}")
                                self.show_message.emit("Test Stopped", 
                                                      f"Agent stuck (limited action diversity) after {steps} steps.",
                                                      False)
                                break
                    
                    # Visualize only if enabled
                    if self.show_visualization:
                        # Visualize (even invalid moves to show what's happening)
                        if disc:
                            self.animate_move_signal.emit(from_rod, to_rod, disc)
                            time.sleep(0.5)
                        else:
                            # Show attempted invalid move
                            time.sleep(0.3)
                        
                        self.update_state_signal.emit([list(rod) for rod in self.env.state])
                        self.update_info_signal.emit({'episode': 1, 'step': steps, 'reward': total_reward})
                    else:
                        # Fast mode - just update info periodically
                        if steps % 10 == 0 or done:
                            self.update_info_signal.emit({'episode': 1, 'step': steps, 'reward': total_reward})
                    
                    state = next_state
                    
                    if done:
                        success = (len(self.env.state[2]) == self.num_discs)
                        print(f"\n=== EPISODE FINISHED ===")
                        print(f"Success: {success}, Steps: {steps}, Total Reward: {total_reward}")
                        if success:
                            self.show_message.emit("Success!", 
                                                  f"Model solved the puzzle in {steps} steps!\nTotal Reward: {total_reward}",
                                                  True)
                        else:
                            self.show_message.emit("Failed", 
                                                  f"Model failed to solve the puzzle in {steps} steps.",
                                                  False)
                        break
                
                if not done and steps >= max_steps:
                    print(f"\n=== TIMEOUT - {steps} STEPS (max: {max_steps}) ===")
                    self.show_message.emit("Timeout", 
                                          f"Model did not solve puzzle in {steps} steps (max: {max_steps}, optimal: {optimal_steps}).",
                                          False)
                
                self.finished.emit()
        
        worker = TestWorker(env, agent, num_discs, show_visualization, exploration, metadata)
        thread = QThread()
        worker.moveToThread(thread)
        
        # Connect signals
        thread.started.connect(worker.run)
        worker.animate_move_signal.connect(visualizer.animate_move)
        worker.update_state_signal.connect(visualizer.update_state)
        worker.update_info_signal.connect(visualizer.update_info)
        worker.show_message.connect(lambda title, msg, success: 
            QMessageBox.information(visualizer, title, msg) if success 
            else QMessageBox.warning(visualizer, title, msg))
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        
        # Store references
        self.test_worker = worker
        self.test_thread = thread
        
        thread.start()
    
    def retest_model(self):
        """Re-run test with the same model."""
        if not hasattr(self, 'test_env') or not hasattr(self, 'test_agent'):
            QMessageBox.warning(self, "No Test Context", "No test context available. Please run a test first.")
            return
        
        # Stop any running test (safely check if thread exists and is valid)
        try:
            if hasattr(self, 'test_thread') and self.test_thread:
                if not sip.isdeleted(self.test_thread) and self.test_thread.isRunning():
                    self.test_thread.quit()
                    self.test_thread.wait(1000)
        except RuntimeError:
            # Thread already deleted, ignore
            pass
        
        # Create new environment and visualizer
        from toh import TowerOfHanoiEnv
        from visualizer import TowerOfHanoiVisualizer
        
        env = TowerOfHanoiEnv(num_discs=self.test_num_discs)
        visualizer = TowerOfHanoiVisualizer(env, num_discs=self.test_num_discs, standalone=False)
        
        # Update visualization page with new visualizer (this sets current_visualizer)
        self.show_visualization_page(visualizer, show_test_again=True)
        
        # Run test episode with stored parameters
        show_viz = getattr(self, 'test_show_visualization', True)
        exploration = getattr(self, 'test_exploration', 0.1)
        metadata = getattr(self, 'test_metadata', None)
        
        self.run_test_episode(env, self.test_agent, visualizer, self.test_num_discs, show_viz, exploration, metadata)
    
    def on_continue_training(self):
        """Load existing model and continue training."""
        from model_selection_dialog import ModelSelectionDialog
        from model_manager import ModelManager
        
        try:
            # Select model to continue training
            dialog = ModelSelectionDialog(self, auto_select_latest=False)
            result = dialog.exec()
            
            if result == QDialog.DialogCode.Accepted:
                model_name, metadata = dialog.get_selected_model()
                if not model_name:
                    QMessageBox.warning(self, "No Selection", "Please select a model to continue training.")
                    return
                
                # Load the model
                model_manager = ModelManager()
                agent, metadata = model_manager.load_model(model_name)
                
                # Show continuation configuration dialog
                continue_config = self.show_continue_training_dialog(metadata)
                if not continue_config:
                    return  # User cancelled
                
                num_discs = continue_config['num_discs']
                episodes = continue_config['episodes']
                keep_epsilon = continue_config['keep_epsilon']
                
                # Handle disc count change
                if num_discs != metadata.get('num_discs', 3):
                    reply = QMessageBox.question(
                        self,
                        "Disc Count Changed",
                        f"⚠️ Original model was trained with {metadata.get('num_discs', 3)} discs.\n"
                        f"You are now training with {num_discs} discs.\n\n"
                        f"The model architecture will be adapted, but performance may vary.\n\n"
                        f"Continue?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return
                    
                    # Recreate agent with new disc count but keep model weights
                    from dqn_agent import DQNAgent
                    new_state_size = num_discs * 3
                    new_agent = DQNAgent(new_state_size, agent.action_size, agent.architecture_name)
                    # Try to transfer weights where possible (first layers)
                    try:
                        for i, layer in enumerate(agent.model.layers):
                            if i < len(new_agent.model.layers):
                                new_agent.model.layers[i].set_weights(layer.get_weights())
                    except:
                        pass  # If shapes don't match, use new weights
                    agent = new_agent
                
                if not keep_epsilon:
                    agent.epsilon = 0.5  # Reset to initial value for exploration
                
                # Show info
                info_msg = f"Continuing training from: {model_name}\n\n"
                info_msg += f"Previous training: {metadata.get('episodes', 0)} episodes\n"
                info_msg += f"Success rate: {metadata.get('success_rate', 0):.1f}%\n"
                info_msg += f"Avg steps: {metadata.get('avg_steps', 0):.1f}\n\n"
                info_msg += f"New training: {episodes} additional episodes\n"
                info_msg += f"Discs: {num_discs}\n"
                info_msg += f"Starting epsilon: {agent.epsilon:.3f}"
                
                QMessageBox.information(self, "Continue Training", info_msg)
                
                # Start training with loaded agent
                from toh import TowerOfHanoiEnv
                from visualizer import TowerOfHanoiVisualizer
                
                env = TowerOfHanoiEnv(num_discs=num_discs)
                visualizer = TowerOfHanoiVisualizer(env, num_discs=num_discs, standalone=False)
                
                # Create performance graph
                self.performance_graph = PerformanceGraphWidget()
                self.performance_graph.set_optimal_steps(num_discs)
                
                # Show visualization page with graph
                self.show_visualization_page(visualizer, performance_graph=self.performance_graph)
                
                config = {
                    'num_discs': num_discs,
                    'episodes': episodes,
                    'show_every': 10,
                    'save_every': 100,
                    'architecture': metadata.get('architecture', 'medium')
                }
                
                # Store agent for later access
                self.current_agent = agent
                
                # Create worker and thread (same as on_train)
                self.training_worker = TrainingWorker(env, agent, visualizer, config)
                self.training_thread = QThread()
                self.training_worker.moveToThread(self.training_thread)
                
                # Connect signals
                self.training_thread.started.connect(self.training_worker.run)
                self.training_worker.progress.connect(self.on_training_progress)
                self.training_worker.finished.connect(self.on_training_finished)
                self.training_worker.model_saved.connect(self.on_model_saved)
                self.training_worker.update_info.connect(visualizer.update_info)
                self.training_worker.update_state.connect(visualizer.update_state)
                self.training_worker.animate_move.connect(visualizer.animate_move)
                self.training_worker.performance_data.connect(self.on_performance_update)
                self.training_worker.finished.connect(self.training_thread.quit)
                
                # Store references for updates
                self.current_visualizer = visualizer
                
                # Start training
                self.training_thread.start()
                
        except Exception as e:
            import traceback
            error_msg = f"Failed to continue training:\n{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
    
    def show_continue_training_dialog(self, metadata):
        """Show dialog to configure continued training."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Continue Training Configuration")
        dialog.setMinimumSize(450, 350)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title = QLabel("Continue Training Configuration")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Model info
        info = QLabel(
            f"Previous training:\n"
            f"  • Episodes: {metadata.get('episodes', 0)}\n"
            f"  • Discs: {metadata.get('num_discs', 3)}\n"
            f"  • Success rate: {metadata.get('success_rate', 0):.1f}%\n"
            f"  • Epsilon: {metadata.get('epsilon', 1.0):.3f}"
        )
        info.setStyleSheet("background-color: #e7f3ff; padding: 10px; border-radius: 5px;")
        layout.addWidget(info)
        
        # Form layout
        form = QFormLayout()
        
        # Number of discs
        discs_spin = QSpinBox()
        discs_spin.setMinimum(3)
        discs_spin.setMaximum(10)
        discs_spin.setValue(metadata.get('num_discs', 3))
        discs_spin.setToolTip("Number of discs to train with (can be different from original)")
        form.addRow("Number of Discs:", discs_spin)
        
        # Additional episodes
        episodes_spin = QSpinBox()
        episodes_spin.setMinimum(100)
        episodes_spin.setMaximum(10000)
        episodes_spin.setSingleStep(100)
        episodes_spin.setValue(500)
        episodes_spin.setToolTip("Additional episodes to train")
        form.addRow("Additional Episodes:", episodes_spin)
        
        # Keep epsilon
        epsilon_check = QCheckBox("Keep current epsilon")
        epsilon_check.setChecked(True)
        epsilon_check.setToolTip("Keep exploration rate or reset to maximum")
        form.addRow("Exploration:", epsilon_check)
        
        layout.addLayout(form)
        
        # Info text
        info_text = QLabel(
            "💡 Tip: Keeping epsilon allows continued learning from current knowledge.\n"
            "Resetting epsilon increases exploration for new patterns."
        )
        info_text.setStyleSheet("color: #666; font-size: 11px; padding: 10px;")
        info_text.setWordWrap(True)
        layout.addWidget(info_text)
        
        layout.addStretch()
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return {
                'num_discs': discs_spin.value(),
                'episodes': episodes_spin.value(),
                'keep_epsilon': epsilon_check.isChecked()
            }
        return None
    
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
            self.training_worker.update_info.connect(visualizer.update_info)
            self.training_worker.update_state.connect(visualizer.update_state)
            self.training_worker.animate_move.connect(visualizer.animate_move)
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
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Form layout for parameters
        form = QFormLayout()
        
        # Model Architecture Selection
        from model_architectures import ModelFactory
        from PyQt6.QtWidgets import QComboBox
        
        self.architecture_combo = QComboBox()
        architectures = ModelFactory.get_all_architectures()
        for arch_name, arch_instance in sorted(architectures.items()):
            info = arch_instance.get_info()
            display_text = f"{arch_name} - {info['complexity']}"
            self.architecture_combo.addItem(display_text, arch_name)
        
        # Set default to Large
        for i in range(self.architecture_combo.count()):
            if "Large (128-64-32)" in self.architecture_combo.itemData(i):
                self.architecture_combo.setCurrentIndex(i)
                break
        
        self.architecture_combo.currentIndexChanged.connect(self.on_architecture_changed)
        
        # Architecture row with preview button
        arch_layout = QHBoxLayout()
        arch_layout.addWidget(self.architecture_combo, stretch=3)
        
        preview_btn = QPushButton("👁️ Preview")
        preview_btn.setToolTip("Preview the neural network architecture")
        preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        preview_btn.clicked.connect(self.preview_architecture)
        arch_layout.addWidget(preview_btn, stretch=1)
        
        form.addRow("Model Architecture:", arch_layout)
        
        # Architecture description
        self.arch_description = QLabel()
        self.arch_description.setWordWrap(True)
        self.arch_description.setStyleSheet("color: #555; font-size: 10px; padding: 5px;")
        form.addRow("", self.arch_description)
        
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
        
        # Update description for initial selection
        self.on_architecture_changed()
        
        # Info label
        info = QLabel(
            "💡 Tip: Larger models learn better but train slower.\n"
            "Start with Large (128-64-32) for best results."
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
    
    def on_architecture_changed(self):
        """Update description when architecture selection changes."""
        from model_architectures import ModelFactory
        
        arch_name = self.architecture_combo.currentData()
        if arch_name:
            info = ModelFactory.get_architecture_info(arch_name)
            desc_text = f"{info['description']}\n"
            desc_text += f"Complexity: {info['complexity']} | "
            desc_text += f"Recommended Episodes: {info['recommended_episodes']}"
            self.arch_description.setText(desc_text)
            
            # Auto-adjust episodes recommendation
            self.episodes_spin.setValue(info['recommended_episodes'])
    
    def preview_architecture(self):
        """Show a preview of the selected architecture."""
        from model_architectures import ModelFactory
        from dqn_agent import DQNAgent
        from model_visualizer import ModelVisualizerWidget
        import numpy as np
        
        arch_name = self.architecture_combo.currentData()
        if not arch_name:
            return
        
        try:
            # Get architecture info
            info = ModelFactory.get_architecture_info(arch_name)
            
            # Create a temporary agent with this architecture
            num_discs = self.num_discs_spin.value()
            state_size = num_discs * 3  # 3 rods, each with num_discs positions
            action_size = 6  # 6 possible moves between 3 rods
            
            temp_agent = DQNAgent(state_size, action_size, architecture_name=arch_name)
            
            # Create preview dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Architecture Preview: {arch_name}")
            dialog.setMinimumSize(900, 650)
            
            layout = QVBoxLayout(dialog)
            
            # Info section
            info_text = f"<h3>{arch_name}</h3>"
            info_text += f"<p><b>Description:</b> {info['description']}</p>"
            info_text += f"<p><b>Complexity:</b> {info['complexity']} | "
            info_text += f"<b>Parameters:</b> {temp_agent.model.count_params():,} | "
            info_text += f"<b>Recommended Episodes:</b> {info['recommended_episodes']}</p>"
            
            info_label = QLabel(info_text)
            info_label.setWordWrap(True)
            info_label.setFont(QFont("Arial", 11))
            info_label.setStyleSheet("padding: 10px; background-color: #f8f9fa; border-radius: 5px;")
            layout.addWidget(info_label)
            
            # Model visualizer
            viz = ModelVisualizerWidget()
            viz_info = f"State Size: {state_size} | Action Size: {action_size} | {temp_agent.model.count_params():,} parameters"
            viz.set_model(temp_agent.model, viz_info)
            layout.addWidget(viz)
            
            # Layer details
            layers_text = "<b>Layer Details:</b><br>"
            for i, layer in enumerate(temp_agent.model.layers):
                layer_type = layer.__class__.__name__
                if hasattr(layer, 'units'):
                    layers_text += f"• Layer {i+1}: {layer_type} - {layer.units} units"
                    if hasattr(layer, 'activation'):
                        layers_text += f" ({layer.activation.__name__})"
                    layers_text += "<br>"
                elif layer_type == 'Dropout':
                    layers_text += f"• Layer {i+1}: Dropout - rate {layer.rate}<br>"
            
            details_label = QLabel(layers_text)
            details_label.setWordWrap(True)
            details_label.setFont(QFont("Arial", 10))
            details_label.setStyleSheet("padding: 10px; background-color: #fff; border: 1px solid #ddd; border-radius: 5px;")
            layout.addWidget(details_label)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    padding: 8px 20px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #5a6268;
                }
            """)
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            
            dialog.exec()
            
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Could not preview architecture:\n{str(e)}")
    
    def get_config(self):
        """Get training configuration."""
        return {
            'architecture': self.architecture_combo.currentData(),
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
