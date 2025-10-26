from gymnasium.spaces import Box, Discrete
import numpy as np

import matplotlib.pyplot as plt
import time


class TowerOfHanoiEnv:
    def __init__(self, num_discs=3):
        self.num_discs = num_discs
        self.state = None  # Initialize the state variable
        self.max_steps = 2 ** num_discs - 1  # Optimal moves for Tower of Hanoi
        self._reset()

        # Define observation space as a Box
        self._observation_space = Box(low=0, high=num_discs, shape=(3, num_discs), dtype=np.int32)
        
        # Define action space (6 possible moves between 3 rods)
        self._action_space = Discrete(6)

    def _reset(self):
        """Reset the environment to the initial state."""
        self.state = [[i for i in range(self.num_discs, 0, -1)], [], []]
        self.steps = 0
        return self.state

    def step(self, action):
        """Apply an action and return the new state, reward, and done flag."""
        from_rod, to_rod = self.decode_action(action)
        reward = -1  # Penalize for each step

        if self.is_valid_move(from_rod, to_rod):
            # Perform the move
            disc = self.state[from_rod].pop()
            self.state[to_rod].append(disc)
            self.steps += 1

            # Check if the game is complete
            if len(self.state[2]) == self.num_discs:
                reward = 100  # Reward for successfully solving
                done = True
            else:
                done = False
        else:
            done = False
            reward = -10  # Heavier penalty for invalid moves

        return self.state, reward, done, {}

    def is_valid_move(self, from_rod, to_rod):
        """Check if a move is valid."""
        if not self.state[from_rod]:  # Can't move from an empty rod
            return False
        if not self.state[to_rod]:  # Can always move to an empty rod
            return True
        return self.state[from_rod][-1] < self.state[to_rod][-1]  # Smaller disc on top

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
        # Plot the state (rods and discs)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(-1, 3)
        ax.set_ylim(0, self.num_discs + 1)

        # Draw rods
        for i in range(3):
            ax.plot([i, i], [0, self.num_discs], color='black', lw=3)

        # Draw discs
        for i in range(3):
            rod = self.state[i]
            for j, disc in enumerate(rod):
                disc_width = disc / self.num_discs * 0.8
                ax.add_patch(plt.Rectangle((i - disc_width/2, j), disc_width, 0.8, 
                                          color=f'C{disc % 10}', lw=2, edgecolor='black'))

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Rod 1', 'Rod 2', 'Rod 3'])
        ax.set_yticks(range(self.num_discs + 1))
        ax.set_title(f'Tower of Hanoi - Step {self.steps}')

        plt.draw()
        plt.pause(0.1)  # Pause to update the plot window

    @property
    def observation_space(self):
        """Return the observation space."""
        return self._observation_space
    
    @property
    def action_space(self):
        """Return the action space."""
        return self._action_space
