"""
Tkinter-based GUI visualizer for Tower of Hanoi with smooth animations.
"""
import tkinter as tk
from tkinter import ttk
import time
import threading


class TowerOfHanoiVisualizer:
    def __init__(self, num_discs=3, animation_speed=0.5):
        """
        Initialize the Tower of Hanoi visualizer.
        
        Args:
            num_discs: Number of discs in the game
            animation_speed: Speed of animation (seconds per move)
        """
        self.num_discs = num_discs
        self.animation_speed = animation_speed
        self.state = [[i for i in range(num_discs, 0, -1)], [], []]
        
        # GUI setup
        self.root = tk.Tk()
        self.root.title("Tower of Hanoi - RL Training Visualization")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Info panel at top
        self.info_frame = tk.Frame(main_frame, bg='#1e1e1e', relief=tk.RAISED, borderwidth=2)
        self.info_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # Episode info
        info_left = tk.Frame(self.info_frame, bg='#1e1e1e')
        info_left.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.episode_label = tk.Label(info_left, text="Episode: 0", font=("Arial", 14, "bold"),
                                      bg='#1e1e1e', fg='#4CAF50')
        self.episode_label.pack(anchor=tk.W)
        
        self.step_label = tk.Label(info_left, text="Steps: 0", font=("Arial", 12),
                                   bg='#1e1e1e', fg='#ffffff')
        self.step_label.pack(anchor=tk.W)
        
        self.reward_label = tk.Label(info_left, text="Total Reward: 0", font=("Arial", 12),
                                     bg='#1e1e1e', fg='#ffffff')
        self.reward_label.pack(anchor=tk.W)
        
        # Training stats
        info_right = tk.Frame(self.info_frame, bg='#1e1e1e')
        info_right.pack(side=tk.RIGHT, padx=20, pady=10)
        
        self.epsilon_label = tk.Label(info_right, text="Epsilon: 1.00", font=("Arial", 12),
                                      bg='#1e1e1e', fg='#FFB74D')
        self.epsilon_label.pack(anchor=tk.E)
        
        self.success_label = tk.Label(info_right, text="Success Rate: 0.0%", font=("Arial", 12),
                                      bg='#1e1e1e', fg='#64B5F6')
        self.success_label.pack(anchor=tk.E)
        
        self.avg_steps_label = tk.Label(info_right, text="Avg Steps: 0.0", font=("Arial", 12),
                                        bg='#1e1e1e', fg='#81C784')
        self.avg_steps_label.pack(anchor=tk.E)
        
        # Canvas for drawing the towers
        self.canvas = tk.Canvas(main_frame, bg='#3a3a3a', highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Control panel at bottom
        control_frame = tk.Frame(main_frame, bg='#1e1e1e', relief=tk.RAISED, borderwidth=2)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        # Speed control
        speed_frame = tk.Frame(control_frame, bg='#1e1e1e')
        speed_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        tk.Label(speed_frame, text="Animation Speed:", font=("Arial", 10),
                bg='#1e1e1e', fg='#ffffff').pack(side=tk.LEFT, padx=(0, 10))
        
        self.speed_var = tk.DoubleVar(value=animation_speed)
        speed_slider = ttk.Scale(speed_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL,
                                length=200, variable=self.speed_var, command=self.update_speed)
        speed_slider.pack(side=tk.LEFT)
        
        self.speed_value_label = tk.Label(speed_frame, text=f"{animation_speed:.2f}s",
                                          font=("Arial", 10), bg='#1e1e1e', fg='#ffffff')
        self.speed_value_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Status
        self.status_label = tk.Label(control_frame, text="Status: Ready", font=("Arial", 10),
                                     bg='#1e1e1e', fg='#9E9E9E')
        self.status_label.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Drawing parameters
        self.canvas_width = 0
        self.canvas_height = 0
        self.rod_positions = []
        self.disc_height = 0
        self.base_y = 0
        
        # Animation state
        self.animating = False
        self.animation_lock = threading.Lock()
        
        # Bind resize event
        self.canvas.bind('<Configure>', self.on_resize)
        
        # Initial draw
        self.root.update()
        self.draw_state()
    
    def on_resize(self, event):
        """Handle canvas resize."""
        self.canvas_width = event.width
        self.canvas_height = event.height
        self.calculate_positions()
        self.draw_state()
    
    def calculate_positions(self):
        """Calculate positions for rods and discs based on canvas size."""
        if self.canvas_width == 0 or self.canvas_height == 0:
            return
        
        # Rod positions (evenly spaced)
        margin = 100
        spacing = (self.canvas_width - 2 * margin) / 2
        self.rod_positions = [
            margin,
            margin + spacing,
            margin + 2 * spacing
        ]
        
        # Disc height
        self.disc_height = max(20, (self.canvas_height - 200) / (self.num_discs + 2))
        self.base_y = self.canvas_height - 100
    
    def update_speed(self, value):
        """Update animation speed."""
        self.animation_speed = float(value)
        self.speed_value_label.config(text=f"{self.animation_speed:.2f}s")
    
    def draw_state(self):
        """Draw the current state of the towers."""
        self.canvas.delete("all")
        
        if self.canvas_width == 0:
            return
        
        # Draw base
        base_margin = 50
        self.canvas.create_rectangle(
            base_margin, self.base_y,
            self.canvas_width - base_margin, self.base_y + 10,
            fill='#1e1e1e', outline='#555555', width=2
        )
        
        # Draw rods
        rod_height = (self.num_discs + 1) * self.disc_height
        for i, x in enumerate(self.rod_positions):
            # Rod
            self.canvas.create_rectangle(
                x - 5, self.base_y - rod_height,
                x + 5, self.base_y,
                fill='#555555', outline='#777777', width=2
            )
            
            # Rod label
            self.canvas.create_text(
                x, self.base_y + 40,
                text=f"Rod {i + 1}",
                font=("Arial", 14, "bold"),
                fill='#ffffff'
            )
        
        # Draw discs
        colors = ['#F44336', '#2196F3', '#4CAF50', '#FFC107', '#9C27B0', '#FF5722', '#00BCD4']
        
        for rod_idx, rod in enumerate(self.state):
            x = self.rod_positions[rod_idx]
            for disc_idx, disc_size in enumerate(rod):
                y = self.base_y - (disc_idx + 1) * self.disc_height
                self.draw_disc(x, y, disc_size, colors[(disc_size - 1) % len(colors)])
    
    def draw_disc(self, x, y, size, color):
        """Draw a single disc."""
        max_width = 150
        width = (size / self.num_discs) * max_width
        height = self.disc_height * 0.8
        
        # Shadow
        self.canvas.create_oval(
            x - width / 2 + 2, y - height / 2 + 2,
            x + width / 2 + 2, y + height / 2 + 2,
            fill='#000000', outline='', stipple='gray50'
        )
        
        # Disc body
        self.canvas.create_oval(
            x - width / 2, y - height / 2,
            x + width / 2, y + height / 2,
            fill=color, outline='#ffffff', width=2
        )
        
        # Disc label
        self.canvas.create_text(
            x, y,
            text=str(size),
            font=("Arial", int(height / 2), "bold"),
            fill='#ffffff'
        )
    
    def animate_move(self, from_rod, to_rod, disc_size):
        """Animate a disc moving from one rod to another."""
        if self.canvas_width == 0:
            return
        
        colors = ['#F44336', '#2196F3', '#4CAF50', '#FFC107', '#9C27B0', '#FF5722', '#00BCD4']
        color = colors[(disc_size - 1) % len(colors)]
        
        # Calculate positions
        from_x = self.rod_positions[from_rod]
        to_x = self.rod_positions[to_rod]
        
        from_disc_idx = len(self.state[from_rod]) - 1
        to_disc_idx = len(self.state[to_rod])
        
        start_y = self.base_y - (from_disc_idx + 1) * self.disc_height
        end_y = self.base_y - (to_disc_idx + 1) * self.disc_height
        lift_y = self.base_y - (self.num_discs + 2) * self.disc_height
        
        # Animation steps
        steps = max(20, int(30 * self.animation_speed))
        
        # Lift up
        for i in range(steps):
            self.canvas.delete("moving_disc")
            t = (i + 1) / steps
            y = start_y + (lift_y - start_y) * t
            self.draw_disc_with_tag(from_x, y, disc_size, color, "moving_disc")
            self.root.update()
            time.sleep(self.animation_speed / steps / 3)
        
        # Move horizontally
        for i in range(steps):
            self.canvas.delete("moving_disc")
            t = (i + 1) / steps
            x = from_x + (to_x - from_x) * t
            self.draw_disc_with_tag(x, lift_y, disc_size, color, "moving_disc")
            self.root.update()
            time.sleep(self.animation_speed / steps / 3)
        
        # Lower down
        for i in range(steps):
            self.canvas.delete("moving_disc")
            t = (i + 1) / steps
            y = lift_y + (end_y - lift_y) * t
            self.draw_disc_with_tag(to_x, y, disc_size, color, "moving_disc")
            self.root.update()
            time.sleep(self.animation_speed / steps / 3)
        
        self.canvas.delete("moving_disc")
    
    def draw_disc_with_tag(self, x, y, size, color, tag):
        """Draw a disc with a specific tag for animation."""
        max_width = 150
        width = (size / self.num_discs) * max_width
        height = self.disc_height * 0.8
        
        # Disc body
        self.canvas.create_oval(
            x - width / 2, y - height / 2,
            x + width / 2, y + height / 2,
            fill=color, outline='#ffffff', width=2,
            tags=tag
        )
        
        # Disc label
        self.canvas.create_text(
            x, y,
            text=str(size),
            font=("Arial", int(height / 2), "bold"),
            fill='#ffffff',
            tags=tag
        )
    
    def update_state(self, new_state, from_rod=None, to_rod=None, animate=True):
        """
        Update the visualizer with a new state.
        
        Args:
            new_state: The new state of the towers
            from_rod: Source rod for the move (for animation)
            to_rod: Destination rod for the move (for animation)
            animate: Whether to animate the move
        """
        with self.animation_lock:
            if animate and from_rod is not None and to_rod is not None:
                # Animate the move
                disc_size = self.state[from_rod][-1]
                self.animate_move(from_rod, to_rod, disc_size)
            
            # Update state
            self.state = [rod[:] for rod in new_state]
            self.draw_state()
            self.root.update()
    
    def update_info(self, episode=None, steps=None, reward=None, epsilon=None,
                   success_rate=None, avg_steps=None, status=None):
        """Update the information display."""
        if episode is not None:
            self.episode_label.config(text=f"Episode: {episode}")
        if steps is not None:
            self.step_label.config(text=f"Steps: {steps}")
        if reward is not None:
            self.reward_label.config(text=f"Total Reward: {reward:.1f}")
        if epsilon is not None:
            self.epsilon_label.config(text=f"Epsilon: {epsilon:.2f}")
        if success_rate is not None:
            self.success_label.config(text=f"Success Rate: {success_rate:.1f}%")
        if avg_steps is not None:
            self.avg_steps_label.config(text=f"Avg Steps: {avg_steps:.1f}")
        if status is not None:
            self.status_label.config(text=f"Status: {status}")
        
        self.root.update()
    
    def reset(self):
        """Reset the visualizer to the initial state."""
        self.state = [[i for i in range(self.num_discs, 0, -1)], [], []]
        self.draw_state()
        self.root.update()
    
    def close(self):
        """Close the visualizer window."""
        try:
            self.root.destroy()
        except:
            pass
    
    def is_closed(self):
        """Check if the window has been closed."""
        try:
            return not self.root.winfo_exists()
        except:
            return True


if __name__ == "__main__":
    # Test the visualizer
    viz = TowerOfHanoiVisualizer(num_discs=3, animation_speed=0.3)
    
    # Simulate some moves
    moves = [
        (0, 2),  # Move disc from rod 0 to rod 2
        (0, 1),  # Move disc from rod 0 to rod 1
        (2, 1),  # Move disc from rod 2 to rod 1
        (0, 2),  # Move disc from rod 0 to rod 2
        (1, 0),  # Move disc from rod 1 to rod 0
        (1, 2),  # Move disc from rod 1 to rod 2
        (0, 2),  # Move disc from rod 0 to rod 2
    ]
    
    viz.update_info(episode=1, steps=0, reward=0, epsilon=1.0,
                   success_rate=0, avg_steps=0, status="Testing")
    
    time.sleep(1)
    
    state = [[3, 2, 1], [], []]
    for i, (from_rod, to_rod) in enumerate(moves):
        # Apply move to state
        disc = state[from_rod].pop()
        state[to_rod].append(disc)
        
        # Update visualization
        viz.update_state(state, from_rod, to_rod, animate=True)
        viz.update_info(steps=i+1, reward=-i-1)
        time.sleep(0.5)
    
    viz.update_info(status="Complete!")
    viz.root.mainloop()
