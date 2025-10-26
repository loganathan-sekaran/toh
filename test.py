import numpy as np
from toh import TowerOfHanoiEnv
from dqn_agent import DQNAgent

from util import flatten_and_reshape


def test_trained_agent(env, agent):
    """Run the trained agent and render its gameplay."""
    state = env._reset()
    state = flatten_and_reshape(state, num_discs, agent)  # Flatten the state
    done = False
    total_steps = 0

    while not done:
        # Render the current state
        env.render()

        # Use the trained model to select an action
        action = np.argmax(agent.model.predict(state, verbose=0)[0])
        next_state, _, done, _ = env.step(action)
        next_state = flatten_and_reshape(next_state, num_discs, agent)  # Flatten the state

        state = next_state
        total_steps += 1

    print(f"Game solved in {total_steps} steps!")
    env.render()  # Show final state


if __name__ == "__main__":
    # Initialize environment and agent
    num_discs = 3
    env = TowerOfHanoiEnv(num_discs=num_discs)
    state_size = np.prod(env.observation_space.shape)  # Flattened state
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    # Test the trained agent
    agent.load("tower_of_hanoi_dqn.h5")
    test_trained_agent(env, agent)
