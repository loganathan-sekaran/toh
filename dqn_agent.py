import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        # Replay memory
        self.memory = deque(maxlen=2000)

        # Neural network for Q-learning
        self.model = self._build_model()

    def _build_model(self):
        """Build a simple neural network."""
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experiences in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action based on the epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])  # Exploit

    def replay(self):
        """Train the model using experiences in replay memory."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

            # Update the Q-value for the chosen action
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            # Train the model
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        """Save the model to a file."""
        self.model.save(name)

    def load(self, name):
        """Load the model from a file."""
        self.model = tf.keras.models.load_model(name)
