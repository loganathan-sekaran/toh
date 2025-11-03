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

        # Hyperparameters - BALANCED for learning (not too aggressive)
        self.gamma = 0.95  # Higher discount for long-term planning
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Slower decay for better exploration
        self.learning_rate = 0.001  # Standard learning rate (was 0.003 - too fast)
        self.batch_size = 32  # Standard batch size
        
        # REDUCED penalty/reward scaling - let environment rewards speak for themselves
        self.penalty_scale = 1.2  # Minimal scaling (was 2.0 - too aggressive)

        # Replay memory
        self.memory = deque(maxlen=10000)  # Larger memory for better learning
        
        # Episode trajectory tracking - MUCH MORE CONSERVATIVE
        self.current_episode_trajectory = []  # Store (state, action, reward) for current episode
        self.trajectory_penalty_propagation = 0.2  # Reduced from 0.5 - less aggressive propagation
        
        # Oscillation detection - LESS aggressive forcing
        self.recent_actions = deque(maxlen=20)  # Track last 20 actions
        self.oscillation_threshold = 8  # Increased from 6 - more patient before forcing
        self.force_random_actions = 0  # Counter: force N random actions when oscillation detected
        self.oscillation_penalty_scale = 1.5  # Reduced from 3.0 - less aggressive scaling

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
        self.current_episode_trajectory = []  # Clear trajectory for new episode

    def remember(self, state, action, reward, next_state, done):
        """
        Store experiences in memory with enhanced temporal credit assignment.
        
        When a large penalty occurs, this method:
        1. Stores the current experience normally
        2. Propagates a fraction of the penalty backwards to recent actions
           that led to this state (helping the agent learn which early 
           choices led to bad outcomes)
        
        This helps the agent learn to "backtrack mentally" during training
        by understanding the consequences of earlier actions.
        """
        # Add to current episode trajectory (for penalty propagation)
        self.current_episode_trajectory.append((state, action, reward, next_state, done))
        
        # Scale penalties (negative rewards) to make them more impactful
        if reward < 0:
            # Check if this is a severe penalty (likely oscillation-related)
            if reward < -10:
                scaled_reward = reward * self.oscillation_penalty_scale
            else:
                scaled_reward = reward * self.penalty_scale
        else:
            scaled_reward = reward
        
        # Store the current experience
        self.memory.append((state, action, scaled_reward, next_state, done))
        
        # TEMPORAL CREDIT ASSIGNMENT: Only for SEVERE penalties (>= -30)
        # And only propagate to immediate previous step to avoid noise
        if reward <= -30 and len(self.current_episode_trajectory) > 1:
            # Only propagate to previous 2 steps (much more conservative)
            propagation_depth = min(2, len(self.current_episode_trajectory) - 1)
            
            for i in range(1, propagation_depth + 1):
                # Get the experience from i steps ago
                idx = -(i + 1)  # -2, -3
                if abs(idx) <= len(self.current_episode_trajectory):
                    prev_state, prev_action, prev_reward, prev_next_state, prev_done = \
                        self.current_episode_trajectory[idx]
                    
                    # Calculate propagated penalty (diminishes with distance)
                    propagation_factor = (self.trajectory_penalty_propagation ** i)
                    propagated_penalty = scaled_reward * propagation_factor
                    
                    # Only propagate if it makes the previous reward worse AND is significant
                    if propagated_penalty < -2:  # Only propagate significant penalties
                        adjusted_reward = prev_reward + (propagated_penalty * 0.2)  # Much smaller fraction
                        
                        # Store the adjusted experience
                        self.memory.append((prev_state, prev_action, adjusted_reward, prev_next_state, prev_done))
                        
                        if i == 1:  # Only print for immediate previous step
                            print(f"  🔗 Credit: {propagated_penalty:.1f} → prev action")
        
        # Clear trajectory if episode ended
        if done:
            self.current_episode_trajectory = []

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
        
        # If oscillation detected, force random exploration for next 5 actions (reduced from 10)
        if oscillation_detected:
            self.force_random_actions = 5
            # Smaller epsilon boost (was 0.3)
            self.epsilon = min(1.0, self.epsilon + 0.1)
        
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
        """
        Train the model using experiences in replay memory with target network.
        Uses BALANCED sampling - not too much focus on penalties.
        """
        if len(self.memory) < self.batch_size:
            return

        # BALANCED EXPERIENCE REPLAY:
        # Sample mostly randomly, with slight bias toward learning from mistakes
        
        # Separate experiences by reward type
        penalty_experiences = []
        neutral_experiences = []
        reward_experiences = []
        
        for exp in self.memory:
            reward = exp[2]  # reward is at index 2
            if reward < -10:  # Significant penalties
                penalty_experiences.append(exp)
            elif reward < 0:  # Minor penalties or neutral
                neutral_experiences.append(exp)
            else:  # Positive rewards
                reward_experiences.append(exp)
        
        # BALANCED composition: 30% penalties, 30% neutral, 40% rewards
        # This ensures agent learns from successes more than failures
        n_penalties = min(int(self.batch_size * 0.3), len(penalty_experiences))
        n_neutral = min(int(self.batch_size * 0.3), len(neutral_experiences))
        n_rewards = self.batch_size - n_penalties - n_neutral
        
        # Sample from each category
        minibatch = []
        
        if len(penalty_experiences) >= n_penalties:
            minibatch.extend(random.sample(penalty_experiences, n_penalties))
        else:
            minibatch.extend(penalty_experiences)
            n_neutral += (n_penalties - len(penalty_experiences))  # Compensate with neutral
        
        if len(neutral_experiences) >= n_neutral:
            minibatch.extend(random.sample(neutral_experiences, n_neutral))
        else:
            minibatch.extend(neutral_experiences)
            n_rewards += (n_neutral - len(neutral_experiences))  # Compensate with rewards
        
        if len(reward_experiences) >= n_rewards:
            minibatch.extend(random.sample(reward_experiences, n_rewards))
        else:
            minibatch.extend(reward_experiences)
            # If still not enough, sample randomly from all
            remaining = self.batch_size - len(minibatch)
            if remaining > 0 and len(self.memory) >= remaining:
                minibatch.extend(random.sample(list(self.memory), remaining))
        
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
