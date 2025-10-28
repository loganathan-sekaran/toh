from gymnasium.spaces import Box, Discrete
import numpy as np

import matplotlib.pyplot as plt
import time


class TowerOfHanoiEnv:
    def __init__(self, num_discs=3):
        self.num_discs = num_discs
        self.state = None  # Initialize the state variable
        self.max_steps = 2 ** num_discs - 1  # Optimal moves for Tower of Hanoi
        
        # Learning enhancements - track performance and patterns
        self.best_steps = float('inf')  # Track best performance across episodes
        self.recent_moves = []  # Track last few moves to detect loops
        self.max_recent_moves = 6  # Window for detecting repetition (increased for sequence detection)
        self.largest_disc = num_discs  # The largest disc number
        self.largest_disc_on_target = False  # Track if largest disc reached target
        self.move_sequence_history = []  # Track sequences of moves for pattern detection
        
        # Progressive placement tracking - which discs are correctly placed on target
        self.correctly_placed_discs = set()  # Set of disc numbers correctly placed on rod 2
        self.current_target_disc = num_discs  # Start with largest disc as target
        
        self._reset()

        # Define observation space as a Box
        self._observation_space = Box(low=0, high=num_discs, shape=(3, num_discs), dtype=np.int32)
        
        # Define action space (6 possible moves between 3 rods)
        self._action_space = Discrete(6)

    def _reset(self):
        """Reset the environment to the initial state."""
        self.state = [[i for i in range(self.num_discs, 0, -1)], [], []]
        self.steps = 0
        self.recent_moves = []  # Clear move history for new episode
        self.largest_disc_on_target = False  # Reset largest disc tracking
        self.move_sequence_history = []  # Clear sequence history for new episode
        self.correctly_placed_discs = set()  # Reset progressive placement tracking
        self.current_target_disc = self.num_discs  # Start with largest disc
        return self.state

    def step(self, action):
        """
        Apply an action and return the new state, reward, and done flag.
        
        Sophisticated reward shaping based on learning logic:
        1. Penalize moving disc on larger disc (invalid move)
        2. Penalize moving largest disc to non-target rod
        3. Reward efficiency - fewer steps than best so far
        4. Penalize repetitive moves (oscillating between rods)
        """
        from_rod, to_rod = self.decode_action(action)
        
        # Store previous state for reward shaping
        prev_discs_on_goal = len(self.state[2])
        
        reward = -0.1  # Small base penalty to encourage efficiency
        done = False

        # Check if move is valid
        if not self.is_valid_move(from_rod, to_rod):
            # LOGIC 1: Moving disc on non-smaller disc (invalid) = HEAVY PENALTY
            reward = -50
            done = False
            return self.state, reward, done, {}
        
        # Valid move - execute it
        disc = self.state[from_rod].pop()
        self.state[to_rod].append(disc)
        self.steps += 1
        
        # Progressive placement reward system
        # Check if disc is placed correctly on target rod
        target_rod = 2  # Rod 3 (index 2) is the target
        
        # Determine which disc should be the current target
        # Target discs in order: largest (num_discs) -> smallest (1)
        expected_on_target = []
        for d in range(self.num_discs, 0, -1):
            if d in self.correctly_placed_discs:
                expected_on_target.append(d)
            else:
                # This is the next disc that needs to be placed
                self.current_target_disc = d
                break
        
        # Check if target rod has correct discs at bottom
        # Target rod should have discs in descending order from bottom
        target_rod_state = self.state[target_rod]
        is_correct_placement = True
        
        if len(target_rod_state) > 0:
            # Verify discs on target are in correct order (largest at bottom)
            for i in range(len(target_rod_state) - 1):
                if target_rod_state[i] <= target_rod_state[i + 1]:
                    is_correct_placement = False
                    break
        
        # Check if we just placed or removed the current target disc
        placed_target_disc = (to_rod == target_rod and disc == self.current_target_disc)
        removed_from_target = (from_rod == target_rod and disc in self.correctly_placed_discs)
        removed_target_disc = (from_rod == target_rod and disc == self.current_target_disc)
        
        # REWARD/PENALTY for progressive placement
        if placed_target_disc and is_correct_placement:
            # Successfully placed the current target disc on target rod!
            # Check if it's placed correctly:
            # - If it's the largest disc, target rod should only have this disc
            # - If it's not the largest, it should be on top of the next larger disc
            correct_position = False
            if disc == self.num_discs:
                # Largest disc - should be only disc on target
                correct_position = (len(target_rod_state) == 1)
            else:
                # Smaller disc - should be on top of next larger disc (disc + 1)
                if len(target_rod_state) >= 2:
                    correct_position = (target_rod_state[-2] == disc + 1) and (disc + 1 in self.correctly_placed_discs)
            
            if correct_position:
                reward += 30  # BIG REWARD for correct placement
                self.correctly_placed_discs.add(disc)
                # Update current target to next smaller disc
                if disc > 1:
                    self.current_target_disc = disc - 1
        
        elif removed_from_target:
            # Removed a disc that was correctly placed - BAD!
            reward -= 40  # HEAVY PENALTY for undoing progress
            self.correctly_placed_discs.discard(disc)
            # Update current target back to this disc
            self.current_target_disc = disc
        
        elif removed_target_disc:
            # Removed the current target disc from target rod before it was locked in
            reward -= 15  # Penalty for removing target disc
        
        # Additional reward for keeping correctly placed discs on target
        for placed_disc in self.correctly_placed_discs:
            if placed_disc in self.state[target_rod]:
                reward += 1  # Small bonus for maintaining correct state
        
        # Track the move for repetition detection (LOGIC 4)
        move = (from_rod, to_rod, disc)
        reverse_move = (to_rod, from_rod, disc)
        
        # Collect all penalties (use max instead of sum to prevent extreme stacking)
        penalties = []
        rewards_bonus = []
        
        # LOGIC 4: Detect repetitive moves (oscillating between same rods)
        if len(self.recent_moves) > 0 and reverse_move == self.recent_moves[-1]:
            penalties.append(10)  # Penalty for immediately reversing last move
        elif self.recent_moves.count(move) >= 2:
            penalties.append(8)  # Penalty for repeating same move multiple times
        
        # NEW LOGIC 5: Detect repeating sequences of multiple moves
        # Add current move to sequence history
        self.move_sequence_history.append(move)
        if len(self.move_sequence_history) > 12:  # Keep last 12 moves for pattern detection
            self.move_sequence_history.pop(0)
        
        # Check for repeating sequences (2-move, 3-move patterns)
        if len(self.move_sequence_history) >= 4:
            # Check for 2-move repeating pattern (A-B-A-B)
            if (len(self.move_sequence_history) >= 4 and
                self.move_sequence_history[-1] == self.move_sequence_history[-3] and
                self.move_sequence_history[-2] == self.move_sequence_history[-4]):
                penalties.append(15)  # Penalty for 2-move cycle
        
        if len(self.move_sequence_history) >= 6:
            # Check for 3-move repeating pattern (A-B-C-A-B-C)
            if (self.move_sequence_history[-1] == self.move_sequence_history[-4] and
                self.move_sequence_history[-2] == self.move_sequence_history[-5] and
                self.move_sequence_history[-3] == self.move_sequence_history[-6]):
                penalties.append(20)  # Penalty for 3-move cycle
        
        # Add to recent moves history
        self.recent_moves.append(move)
        if len(self.recent_moves) > self.max_recent_moves:
            self.recent_moves.pop(0)
        
        # NEW LOGIC 6: Reward keeping largest disc on target, penalize removing it
        was_largest_on_target = self.largest_disc_on_target
        is_largest_on_target = self.largest_disc in self.state[2]
        
        if is_largest_on_target and not was_largest_on_target:
            # Just placed largest disc on target
            rewards_bonus.append(25)  # Big reward for getting it there
            self.largest_disc_on_target = True
        elif not is_largest_on_target and was_largest_on_target:
            # Removed largest disc from target - BAD!
            penalties.append(30)  # Heavy penalty for undoing progress
            self.largest_disc_on_target = False
        elif is_largest_on_target and was_largest_on_target:
            # Largest disc still on target - good!
            rewards_bonus.append(2)  # Small bonus for maintaining correct state
        
        # LOGIC 2: Largest disc strategy (original logic - now works with Logic 6)
        if disc == self.largest_disc:
            if to_rod == 2:
                rewards_bonus.append(20)  # BIG reward for moving largest disc to target!
            else:
                penalties.append(15)  # Penalty for moving largest disc to wrong rod
        
        # Apply the MAXIMUM penalty instead of sum (prevents extreme stacking)
        if penalties:
            reward -= max(penalties)
        # Add all reward bonuses (positive stacking is OK)
        if rewards_bonus:
            reward += sum(rewards_bonus)
        
        # Reward shaping: bonus for moving discs to goal rod
        new_discs_on_goal = len(self.state[2])
        if new_discs_on_goal > prev_discs_on_goal:
            reward += 5  # Bonus for moving a disc to the goal
            # Extra bonus if disc is in correct position (larger discs should be at bottom)
            if to_rod == 2 and disc == max(self.state[2]):
                reward += 3  # Correct placement bonus
        
        # Check if puzzle is complete
        if len(self.state[2]) == self.num_discs:
            # LOGIC 3: Reward based on efficiency compared to best performance
            optimal_steps = self.max_steps
            
            # Compare with best previous performance
            if self.steps < self.best_steps:
                # New best! Huge reward
                efficiency_bonus = (self.best_steps - self.steps) * 20
                reward += 100 + efficiency_bonus
                self.best_steps = self.steps  # Update best
            elif self.steps == optimal_steps:
                # Optimal solution!
                reward += 200
                self.best_steps = self.steps
            elif self.steps <= optimal_steps * 1.5:
                # Good solution (within 50% of optimal)
                reward += 80
            else:
                # Completed but inefficient
                reward += 50
            
            done = True
        
        # Clip reward to prevent numerical instability
        reward = np.clip(reward, -100, 300)
        
        return self.state, reward, done, {}

    def is_valid_move(self, from_rod, to_rod):
        """Check if a move is valid."""
        if not self.state[from_rod]:  # Can't move from an empty rod
            return False
        if not self.state[to_rod]:  # Can always move to an empty rod
            return True
        return self.state[from_rod][-1] < self.state[to_rod][-1]  # Smaller disc on top
    
    def get_valid_actions(self):
        """Get list of valid action indices in current state."""
        valid_actions = []
        for action in range(self.action_space.n):
            from_rod, to_rod = self.decode_action(action)
            if self.is_valid_move(from_rod, to_rod):
                valid_actions.append(action)
        return valid_actions

    def decode_action(self, action):
        """Decode an action into (from_rod, to_rod)."""
        # Example: With 3 rods, actions 0-5 correspond to moves:
        # 0: rod 0 -> rod 1
        # 1: rod 0 -> rod 2
        # 2: rod 1 -> rod 0
        # 3: rod 1 -> rod 2
        # 4: rod 2 -> rod 0
        # 5: rod 2 -> rod 1
        from_rod = action // (len(self.state) - 1)
        to_rod = action % (len(self.state) - 1)
        if to_rod >= from_rod:
            to_rod += 1
        return from_rod, to_rod

    def render(self, mode='human'):
        """Visualize the rods and discs using Matplotlib."""
        # Reuse existing figure or create new one
        if not hasattr(self, '_fig') or not plt.fignum_exists(self._fig.number):
            self._fig, self._ax = plt.subplots(figsize=(6, 4))
            plt.ion()  # Enable interactive mode
        else:
            self._ax.clear()
        
        # Plot the state (rods and discs)
        self._ax.set_xlim(-1, 3)
        self._ax.set_ylim(0, self.num_discs + 1)

        # Draw rods
        for i in range(3):
            self._ax.plot([i, i], [0, self.num_discs], color='black', lw=3)

        # Draw discs
        for i in range(3):
            rod = self.state[i]
            for j, disc in enumerate(rod):
                disc_width = disc / self.num_discs * 0.8
                self._ax.add_patch(plt.Rectangle((i - disc_width/2, j), disc_width, 0.8, 
                                          color=f'C{disc % 10}', lw=2, edgecolor='black'))

        self._ax.set_xticks([0, 1, 2])
        self._ax.set_xticklabels(['Rod 1', 'Rod 2', 'Rod 3'])
        self._ax.set_yticks(range(self.num_discs + 1))
        self._ax.set_title(f'Tower of Hanoi - Step {self.steps}')

        plt.draw()
        plt.pause(0.01)  # Brief pause to update the plot window

    @property
    def observation_space(self):
        """Return the observation space."""
        return self._observation_space
    
    @property
    def action_space(self):
        """Return the action space."""
        return self._action_space
