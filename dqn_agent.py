import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
from model_architectures import ModelFactory


class DQNAgent:
    def __init__(self, state_size, action_size, architecture_name="Medium (64-32)"):
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

        # Hyperparameters - optimized for Tower of Hanoi with strong reward shaping
        self.gamma = 0.90  # Lower discount to prioritize immediate rewards/penalties
        self.epsilon = 0.5  # Start with 50% exploitation (was 1.0 - too much random exploration)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # Faster decay to reach min in ~100 episodes (was 0.995)
        self.learning_rate = 0.003  # Increased for faster learning from penalties (was 0.001)
        self.batch_size = 64  # Larger batch for more stable learning (was 32)
        self.penalty_scale = 2.0  # Scale penalties by this factor when storing in memory

        # Replay memory
        self.memory = deque(maxlen=10000)  # Larger memory for better learning
        
        # Oscillation detection - track recent actions and FORCE exploration to break loops
        self.recent_actions = deque(maxlen=20)  # Track last 20 actions
        self.oscillation_threshold = 6  # If same pattern repeats in 6 actions, break it
        self.force_random_actions = 0  # Counter: force N random actions when oscillation detected
        self.oscillation_penalty_scale = 3.0  # Extra penalty scaling for oscillation experiences

        # Get the model architecture
        self.architecture = ModelFactory.get_architecture(architecture_name)
        
        # Neural network for Q-learning (main network)
        self.model = self._build_model()
        
        # Target network for stable training
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Target network update frequency
        self.target_update_counter = 0
        self.target_update_freq = 100  # Update target every 100 training steps (was 10)

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
    
    def reset_episode(self):
        """Reset episode-specific tracking (call at start of each episode)."""
        self.recent_actions.clear()

    def remember(self, state, action, reward, next_state, done):
        """
        Store experiences in memory.
        Applies penalty scaling to amplify negative rewards for faster learning.
        Extra scaling for oscillation-related penalties.
        """
        # Scale penalties (negative rewards) to make them more impactful
        if reward < 0:
            # Check if this is a severe penalty (likely oscillation-related)
            if reward < -10:
                scaled_reward = reward * self.oscillation_penalty_scale
            else:
                scaled_reward = reward * self.penalty_scale
        else:
            scaled_reward = reward
        
        self.memory.append((state, action, scaled_reward, next_state, done))

    def act(self, state, valid_actions=None):
        """
        Choose an action based on the epsilon-greedy policy.
        Includes oscillation detection with FORCED exploration to break loops.
        
        Args:
            state: Current state
            valid_actions: Optional list of valid action indices. If provided, only these actions will be considered.
            
        Returns:
            action: The selected action index
        """
        # Check for oscillation patterns and FORCE exploration when detected
        oscillation_detected = False
        
        if len(self.recent_actions) >= self.oscillation_threshold:
            last_actions = list(self.recent_actions)[-self.oscillation_threshold:]
            
            # Check for repetitive single action: A,A,A,A,A,A...
            if len(set(last_actions)) == 1:
                oscillating_action = last_actions[0]
                print(f"⚠️  REPETITIVE ACTION: action {oscillating_action} repeated {self.oscillation_threshold} times - FORCING EXPLORATION")
                oscillation_detected = True
            
            # Check for alternating 2-action pattern: A,B,A,B,A,B...
            elif len(set(last_actions)) == 2:
                is_alternating = all(
                    last_actions[i] != last_actions[i+1] 
                    for i in range(len(last_actions)-1)
                )
                if is_alternating:
                    print(f"⚠️  OSCILLATION: 2-action cycle detected - FORCING EXPLORATION")
                    oscillation_detected = True
            
            # Check for 3-action cycle: A,B,C,A,B,C...
            elif len(set(last_actions)) == 3 and len(last_actions) >= 6:
                mid = len(last_actions) // 2
                if last_actions[:mid] == last_actions[mid:mid*2]:
                    print(f"⚠️  3-ACTION CYCLE: pattern repeating - FORCING EXPLORATION")
                    oscillation_detected = True
        
        # If oscillation detected, force random exploration for next 10 actions
        if oscillation_detected:
            self.force_random_actions = 10
            # Also boost epsilon temporarily to encourage more exploration
            self.epsilon = min(1.0, self.epsilon + 0.3)
        
        # Force random action if counter is active OR normal epsilon-greedy
        if self.force_random_actions > 0 or np.random.rand() <= self.epsilon:
            if self.force_random_actions > 0:
                self.force_random_actions -= 1
                
            # Explore: choose random action (only from valid actions if provided)
            if valid_actions is not None and len(valid_actions) > 0:
                action = random.choice(valid_actions)
                self.recent_actions.append(action)
                return action
            action = random.randrange(self.action_size)
            self.recent_actions.append(action)
            return action
        
        # Exploit: choose best action
        q_values = self.model.predict(state, verbose=0)[0]
        
        # If valid actions are provided, mask out invalid actions
        if valid_actions is not None and len(valid_actions) > 0:
            # Set Q-values of invalid actions to very negative number
            masked_q_values = np.full(self.action_size, -np.inf)
            for action in valid_actions:
                masked_q_values[action] = q_values[action]
            
            action = np.argmax(masked_q_values)
            self.recent_actions.append(action)
            return action
        
        action = np.argmax(q_values)
        self.recent_actions.append(action)
        return action

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
