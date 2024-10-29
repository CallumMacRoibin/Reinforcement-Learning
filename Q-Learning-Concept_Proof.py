import numpy as np
import random
import matplotlib.pyplot as plt

# Hyperparameters
loops1 = 300
loops2 = 30
learning_rate = 0.15
discount_factor = 0.95
epsilon = 1     # Start value

# Initialize Q-matrix
q_matrix = np.zeros((49, 4))

# Define valid actions and transitions
valid_actions = np.array([
    [1, 3], [0, 1, 3], [0, 3], [1, 2, 3], [0, 1, 2, 3], [0, 2, 3],
    [1, 2, 3], [0, 1, 2, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3], [0, 2, 3],
    [1, 2, 3], [0, 1, 2, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3], [0, 2, 3],
    [1, 2], [0, 1, 2], [0, 2]
])

transition_matrix = np.array([
    [-1, 1, -1, 3], [0, 2, -1, 4], [1, -1, -1, 5], [-1, 4, 0, 6],
    [3, 5, 1, 7], [4, -1, 2, 8], [-1, 7, 3, 9], [6, 8, 4, 10],
    [7, -1, 5, 11], [-1, 10, 6, 12], [9, 11, 7, 13], [10, -1, 8, 14],
    [-1, 13, 9, 15], [12, 14, 10, 16], [13, -1, 11, 17], [-1, 16, 12, 18],
    [15, 17, 13, 19], [16, -1, 14, 20], [-1, 19, 15, -1], [18, 20, 16, -1],
    [19, -1, 17, -1]
])

reward = np.array([
    [-1000, -0.125, -1000, 0], [0, -0.375, -1000, 0], [0, -1000, -1000, 1.125],
    [-1000, -0.25, -0.25, -0.375], [0, 1.25, -1.5, 0], [0.125, -1000, -1.125, 2.375],
    [-1000, -0.375, -0.25, -0.125], [0, 1.375, -0.75, 0], [-1.375, -1000, -2.25, 2.75],
    [-1000, -0.375, -0.25, -0.375], [-0.125, 1.625, 0.25, 0], [-2.125, -1000, -1.875, 2.25],
    [-1000, 0, 0, 0], [0, 1.875, -0.25, 0.125], [-2.25, -1000, -2.375, 2.5],
    [-1000, 0.25, -0.375, 0], [0, 2.125, -6.875, 0], [-3.875, -1000, -1.25, -5.125],
    [-1000, 0, 0, -1000], [0, 0.125, -1.875, -1000], [0.25, -1000, 0.25, -1000]
])

# Training phase
for episode in range(loops1):
    start_state = 9
    current_state = start_state
    print(f"Iteration number {episode}, Epsilon {epsilon}!")

    for _ in range(loops2):
        print(current_state)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(valid_actions[current_state])
        else:
            action = np.argmax(q_matrix[current_state, :])

        # Transition to the next state
        next_state = transition_matrix[current_state][action]
        future_rewards = [q_matrix[next_state][action_nxt] for action_nxt in valid_actions[next_state]]
        old_q = q_matrix[current_state][action]
        new_q = reward[current_state][action] + discount_factor * max(future_rewards)
        q_matrix[current_state][action] = old_q + (learning_rate * (new_q - old_q))
        current_state = next_state

    # Decay epsilon after each episode
    if epsilon > 0:
        epsilon -= 1 / loops1
    else:
        epsilon = 0

print("Training Over")

# Testing phase
current_state = 9
actions = []
states = [current_state]

for _ in range(loops2):
    step = np.argmax(q_matrix[current_state, :])
    if step == 0:
        print("go left")
        current_state -= 1
    elif step == 1:
        print("go right")
        current_state += 1
    elif step == 2:
        print("go up")
        current_state += 3
    elif step == 3:
        print("go down")
        current_state -= 3
    actions.append(step)
    states.append(current_state)

print("Actions Taken:", actions)
print("States:", states)
