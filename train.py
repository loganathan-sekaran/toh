import numpy as np
from toh import TowerOfHanoiEnv
from dqn_agent import DQNAgent
from util import flatten_and_reshape
import matplotlib.pyplot as plt

def train_tower_of_hanoi(env, agent, episodes=500):
    for e in range(episodes):
        state = env._reset()
        state = flatten_and_reshape(state, num_discs, agent)  # Flatten the state
        total_reward = 0

        for time in range(500):  # Maximum steps per episode
            # Render the environment every 10 episodes
            if e % 10 == 0:
                env.render()

            # Agent selects an action
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = flatten_and_reshape(next_state, num_discs, agent)

            # Remember experience
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                print(
                    f"Episode: {e + 1}/{episodes}, Steps: {time + 1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        # Train the agent with replay memory
        agent.replay()

    # Save the trained model
    agent.save("tower_of_hanoi_dqn.h5")


if __name__ == "__main__":
    # Initialize environment and agent
    num_discs = 3
    env = TowerOfHanoiEnv(num_discs=num_discs)
    state_size = np.prod(env.observation_space.shape)  # Flattened state
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    # Train the agent
    train_tower_of_hanoi(env, agent, episodes=500)

    # Keep the plot open after training
    plt.show()





