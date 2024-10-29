# Import required modules
import numpy as np
import random
import matplotlib.pyplot as plt
import math

# Initialize parameters
mult = 0
start_state = 21
loops1 = 1000
loops2 = 30
discount_factor = 0.94
learning_rate = 0.15
epsilon_value = 1  # Start value for epsilon

# Initialize Q-matrix with zeros
q_matrix = np.zeros((49, 4))

# Epsilon decay parameters
A, B, C = 0.9, 0.15, 0.1

# Epsilon decay function
def epsilon(itt):
    exp_value = (itt - A * loops1) / (B * loops1)
    cosh = np.cosh(math.exp(-exp_value))
    epsilon = 1 - (1 / cosh + (itt * C / loops1))
    return epsilon

# Define valid actions and state transitions
valid_actions = np.array([
    [1, 3], [0, 1, 3], [0, 1, 3], [0, 1, 3], [0, 1, 3], [0, 1, 3], [0, 3],
    # More valid actions can go here...
])

transition_matrix = np.array([
    [-1, 1, -1, 7], [0, 2, -1, 8], [1, 3, -1, 9],
    # More transitions can go here...
])

# Define rewards for each state-action pair
change = np.array([
    [-1000, 0, -1000, 0], [0, 0, -1000, 0],
    # More rewards can go here...
])

# Coordinates for visualization
co_ordinate = np.array([
    [11.47831871, 6.103130316], [12.55703574, 3.364647586],
    # More coordinates can go here...
])

# Main training loop
for i in range(loops1):
    loop_number = i
    current_state = start_state
    print(f"Iteration number {i}, Epsilon {epsilon_value}!")

    for _ in range(loops2):
        # Epsilon-Greedy Strategy for action selection
        if random.uniform(0, 1) < epsilon_value:
            action = random.choice(valid_actions[current_state])
        else:
            q_values = q_matrix[current_state, :]
            q_values_sorted = sorted(q_values, reverse=True)
            maximum = q_values_sorted[0]
            if maximum == 0:
                maximum = q_values_sorted[1] if q_values_sorted[1] else q_values_sorted[2]
                if maximum == 0:
                    maximum = q_values_sorted[3]

            action = np.argmax(q_values == maximum)

        # Get the reward for the selected action
        change_value = change[current_state][action]
        reward = 0.3 if change_value == 0 else change_value

        # Q-learning update
        next_state = transition_matrix[current_state][action]
        future_rewards = [q_matrix[next_state][a] for a in valid_actions[next_state]]
        old_q = q_matrix[current_state][action]
        new_q = reward + discount_factor * max(future_rewards) - old_q
        q_matrix[current_state][action] += learning_rate * new_q
        current_state = next_state

    # Update epsilon value with decay
    epsilon_value = epsilon(loop_number) if epsilon_value > 0 else 0

print("Training Over")

# Policy evaluation
current_state = start_state
actions, states = [], [current_state]

# Follow the policy to find optimal path
for _ in range(loops2):
    q_values = q_matrix[current_state, :]
    maximum = max(q_values)
    if maximum == 0:
        maximum = next(val for val in q_values if val != 0)

    step = np.argmax(q_values == maximum)
    action_text = ["left", "right", "up", "down"]
    print(f"go {action_text[step]}")

    # Update state based on action
    if step == 0:
        current_state -= 1
    elif step == 1:
        current_state += 1
    elif step == 2:
        current_state -= 7
    elif step == 3:
        current_state += 7

    actions.append(step)
    states.append(current_state)

# Convert states to coordinates for plotting
states_list = np.array(states).reshape(-1, 1)
x_co_ordinates, y_co_ordinates = [], []

for i in range(loops2):
    state_no = states_list[i][0]
    x_co_ordinates.append(co_ordinate[state_no][0])
    y_co_ordinates.append(co_ordinate[state_no][1])

# Plot path
plt.plot(x_co_ordinates, y_co_ordinates)
plt.ylabel('cm')
plt.xlabel('cm')
plt.title('Movements')
plt.show()

# Output the actions and states
print("Actions Taken:", actions)
print("States:", states)
