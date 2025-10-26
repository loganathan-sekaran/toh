import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
from model_architectures import ModelFactory


class DQNAgent:
    def __init__(self, state_size, action_size, architecture_name="Large (128-64-32)"):
        """
        Initialize DQN Agent with a specific model architecture.
        
        Args:
            state_size: Size of the input state
            action_size: Number of possible actions
            architecture_name: Name of the model architecture to use (see ModelFactory.list_architectures())
        """
        self.state_size = state_size
        self.action_size = action_size
        self.architecture_name = architecture_name

        # Hyperparameters - optimized for Tower of Hanoi
        self.gamma = 0.99  # Discount factor (higher for long-term planning)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005  # Lower learning rate for stability
        self.batch_size = 64

        # Replay memory
        self.memory = deque(maxlen=10000)  # Larger memory for better learning

        # Get the model architecture
        self.architecture = ModelFactory.get_architecture(architecture_name)
        
        # Neural network for Q-learning (main network)
        self.model = self._build_model()
        
        # Target network for stable training
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Target network update frequency
        self.target_update_counter = 0
        self.target_update_freq = 10  # Update target every 10 training steps

    def _build_model(self):
        """Build neural network using the selected architecture."""
        return self.architecture.build(
            self.state_size, 
            self.action_size, 
            self.learning_rate
        )
    
    def update_target_model(self):
        """Copy weights from main model to target model."""
        self.target_model.set_weights(self.model.get_weights())

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
        """Train the model using experiences in replay memory with target network."""
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch arrays
        states = np.array([experience[0][0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3][0] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Batch predict current Q-values
        current_q_values = self.model.predict(states, verbose=0)
        
        # Batch predict next Q-values using target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Calculate target Q-values
        target_q_values = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                # Use target network for stable Q-value estimation
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])
        
        # Train the model on the entire batch at once
        self.model.fit(states, target_q_values, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Update target network periodically
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.update_target_model()
            self.target_update_counter = 0

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        """Save the model to a file."""
        self.model.save(name)

    def load(self, name):
        """Load the model from a file."""
        self.model = tf.keras.models.load_model(name)
