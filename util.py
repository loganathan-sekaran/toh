import numpy as np

def flatten_state(state, num_discs):
    # Flatten the state: Each rod has discs, and empty rods are represented by zeros.
    flattened_state = []
    for rod in state:
        # Represent discs on the rod from top to bottom
        rod_state = rod + [0] * (num_discs - len(rod))  # Fill with 0 for empty discs
        flattened_state.extend(rod_state)  # Add rod state to the flattened list
    return np.array(flattened_state)

def flatten_and_reshape(state, num_discs, agent):
    state = flatten_state(state, num_discs)
    state = np.reshape(state, [1, agent.state_size])
    return state


