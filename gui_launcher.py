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


class TrainingWorker(QObject):
    """Worker object for running training in a separate thread."""
    finished = pyqtSignal(str)  # Emits completion message
    progress = pyqtSignal(int, float, float)  # episode, epsilon, success_rate
    model_saved = pyqtSignal(str, object)  # model_path, metadata
    update_info = pyqtSignal(dict)  # GUI update data (episode, step, reward, epsilon)
    update_state = pyqtSignal(object)  # Environment state update
    animate_move = pyqtSignal(int, int, int)  # from_rod, to_rod, disc
    
    def __init__(self, env, agent, visualizer, config):
        super().__init__()
        self.env = env
        self.agent = agent
        self.visualizer = visualizer
        self.config = config
        self.should_stop = False
        self.success_count = 0
        self.total_steps = 0
    
    def run(self):
        """Run the training loop."""
        from util import flatten_state
        import numpy as np
        from model_manager import ModelManager
        
        state_size = np.prod(self.env.observation_space.shape)
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
            state_size = np.prod(env.observation_space.shape)
            action_size = env.action_space.n
            
            # Create agent with selected architecture
            agent = DQNAgent(state_size, action_size, architecture_name=config['architecture'])
            
            # Create visualizer and show it
            visualizer = TowerOfHanoiVisualizer(env, num_discs=config['num_discs'], standalone=False)
            self.show_visualization_page(visualizer)
            
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
            self.training_worker.finished.connect(self.training_thread.quit)
            
            # Store references for updates
            self.current_visualizer = visualizer
            
            # Start training
            self.training_thread.start()
    
    def on_training_progress(self, episode, epsilon, success_rate):
        """Handle training progress updates (runs on main thread)."""
        if hasattr(self, 'current_visualizer'):
            self.current_visualizer.update_info({'episode': episode, 'epsilon': epsilon, 'success_rate': success_rate})
            QApplication.processEvents()
    
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
        
        dialog = ModelSelectionDialog(self, auto_select_latest=True)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            model_name, metadata = dialog.get_selected_model()
            if model_name:
                # Load the model
                model_manager = ModelManager()
                try:
                    agent, metadata = model_manager.load_model(model_name)
                    
                    # Show model architecture
                    self.show_model_architecture(agent.model, metadata)
                    
                    # Run test with visualization
                    from toh import TowerOfHanoiEnv
                    from visualizer import TowerOfHanoiVisualizer
                    from util import flatten_state
                    import numpy as np
                    
                    num_discs = metadata.get('num_discs', 3)
                    env = TowerOfHanoiEnv(num_discs=num_discs)
                    visualizer = TowerOfHanoiVisualizer(env, num_discs=num_discs, standalone=False)
                    
                    self.show_visualization_page(visualizer)
                    
                    # Run test episode
                    self.run_test_episode(env, agent, visualizer, num_discs)
                    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
    
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
    
    def run_test_episode(self, env, agent, visualizer, num_discs):
        """Run a single test episode with visualization"""
        from util import flatten_state
        import numpy as np
        
        class TestWorker(QObject):
            animate_move_signal = pyqtSignal(int, int, int)
            update_state_signal = pyqtSignal(object)
            update_info_signal = pyqtSignal(dict)
            show_message = pyqtSignal(str, str, bool)  # title, message, success
            finished = pyqtSignal()
            
            def __init__(self, env, agent, num_discs):
                super().__init__()
                self.env = env
                self.agent = agent
                self.num_discs = num_discs
            
            def run(self):
                import time
                state = self.env._reset()
                self.update_state_signal.emit(self.env.state)
                self.update_info_signal.emit({'episode': 1, 'step': 0, 'reward': 0})
                time.sleep(1)
                
                done = False
                steps = 0
                total_reward = 0
                
                while not done and steps < 100:
                    flat_state = flatten_state(state, self.num_discs)
                    flat_state = np.reshape(flat_state, [1, self.agent.state_size])
                    
                    # Use agent to select action (no exploration)
                    q_values = self.agent.model.predict(flat_state, verbose=0)
                    action = np.argmax(q_values[0])
                    
                    # Get move details
                    from_rod, to_rod = self.env.decode_action(action)
                    disc = self.env.state[from_rod][-1] if self.env.state[from_rod] else None
                    
                    # Execute action
                    next_state, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    # Visualize
                    if disc:
                        self.animate_move_signal.emit(from_rod, to_rod, disc)
                        time.sleep(0.3)
                        self.update_state_signal.emit([list(rod) for rod in self.env.state])
                        self.update_info_signal.emit({'episode': 1, 'step': steps, 'reward': total_reward})
                    
                    state = next_state
                    
                    if done:
                        success = (len(self.env.state[2]) == self.num_discs)
                        if success:
                            self.show_message.emit("Success!", 
                                                  f"Model solved the puzzle in {steps} steps!",
                                                  True)
                        else:
                            self.show_message.emit("Failed", 
                                                  f"Model failed to solve the puzzle in {steps} steps.",
                                                  False)
                        break
                
                self.finished.emit()
        
        worker = TestWorker(env, agent, num_discs)
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
