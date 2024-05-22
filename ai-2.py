import numpy as np

# Define the grid size and rewards
grid_size = 3
rewards = np.zeros((grid_size, grid_size))
rewards[2, 2] = 1  # Goal state

# Define the discount factor
gamma = 0.9

# Define possible actions
actions = ['up', 'down', 'left', 'right']
action_effects = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Function to get the next state and reward
def step(state, action):
    next_state = (state[0] + action_effects[action][0], state[1] + action_effects[action][1])
    if next_state[0] < 0 or next_state[0] >= grid_size or next_state[1] < 0 or next_state[1] >= grid_size:
        next_state = state
    reward = rewards[next_state]
    return next_state, reward

# Initialize value function
values = np.zeros((grid_size, grid_size))

# Value Iteration algorithm
def value_iteration():
    threshold = 1e-4
    while True:
        delta = 0
        new_values = np.copy(values)
        for i in range(grid_size):
            for j in range(grid_size):
                value_list = []
                for action in actions:
                    (next_i, next_j), reward = step((i, j), action)
                    value_list.append(reward + gamma * values[next_i, next_j])
                new_values[i, j] = max(value_list)
                delta = max(delta, abs(values[i, j] - new_values[i, j]))
        values[:] = new_values
        if delta < threshold:
            break

# Run value iteration
value_iteration()
print("Optimal Value Function:")
print(values)
