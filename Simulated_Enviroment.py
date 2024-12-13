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
[1,3], 
[0,1,3], 
[0,1,3], 
[0,1,3], 
[0,1,3], 
[0,1,3], 
[0,3], 
[1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,2,3], 
[1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,2,3], 
[1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,2,3], 
[1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,2,3], 
[1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,1,2,3], 
[0,2,3], 
[1,2], 
[0,1,2], 
[0,1,2], 
[0,1,2], 
[0,1,2], 
[0,1,2], 
[0,2] 
],dtype=object)

transition_matrix = np.array([
[-1,1,-1,7], 
[0,2,-1,8], 
[1,3,-1,9], 
[2,4,-1,10], 
[3,5,-1,11], 
[4,6,-1,12], 
[5,-1,-1,13], 
[-1,8,0,14], 
[7,9,1,15], 
[8,10,2,16], 
[9,11,3,17], 
[10,12,4,18], 
[11,13,5,19], 
[12,-1,6,20], 
[-1,15,7,21], 
[14,16,8,22], 
[15,17,9,23], 
[16,18,10,24], 
[17,19,11,25], 
[18,20,12,26], 
[19,-1,13,27], 
[-1,22,14,28], 
[21,23,15,29], 
[22,24,16,30], 
[23,25,17,31], 
[24,26,18,32], 
[25,27,19,33], 
[26,-1,20,34], 
[-1,29,21,35], 
[28,30,22,36], 
[29,31,23,37], 
[30,32,24,38], 
[31,33,25,39], 
[32,34,26,40], 
[33,-1,27,41], 
[-1,36,28,42], 
[35,37,29,43], 
[36,38,30,44], 
[37,39,31,45], 
[38,40,32,46], 
[39,41,33,47], 
[40,-1,34,48], 
[-1,43,35,-1], 
[42,44,36,-1], 
[43,45,37,-1], 
[44,46,38,-1], 
[45,47,39,-1], 
[46,48,40,-1], 
[47,-1,41,-1] 
])

# Define rewards for each state-action pair
change = np.array([
[-1000,0,-1000,0], 
[0,0,-1000,0], 
[0,0,-1000,0], 
[0,0,-1000,0], 
[0,0,-1000,0], 
[-0.1,0.3,-1000,1.1], 
[-1.4,-1000,-1000,1.4], 
[-1000,0,0,0], 
[0,0,0,0], 
[0,0,0,0], 
[0,0,0,0], 
[0,0.7,-0.1,0.6], 
[-1.3,0.9,-1.1,1.3], 
[-2.5,-1000,-1.3,1.8], 
[-1000,0,0,0], 
[0,0,0,0], 
[0,0,0,0], 
[0,0,0,0], 
[-0.8,0.5,-0.8,1.2], 
[-2,0.7,-1.8,1.3], 
[-2.1,-1000,-1.6,1.1], 
[-1000,0,0,0], 
[0,0,0,0], 
[0,0,0,0], 
[-0.2,1,-0.3,0.2], 
[-1,0.9,-1.7,1.2], 
[-1.9,0.9,-1.6,1.1], 
[-2.3,-1000,-1.7,0.8], 
[-1000,0,0,0], 
[0,0,0,0], 
[0,0,0,0], 
[-0.2,1,-1,0.9], 
[-1.6,1,-1.9,0.9], 
[-1.9,1.7,-1.9,0.3], 
[-1.5,-1000,-1.4,-0.8], 
[-1000,0,0,0], 
[0,0,0,0], 
[0,0.2,-0.1,0], 
[-0.2,1.2,-1.3,0.4], 
[-1.3,0.9,-1.1,-0.1], 
[-1.3,0.9,-0.1,-1.2], 
[-0.4,-1000,0.1,-0.3], 
[-1000,0,0,-1000], 
[0,0,0,-1000], 
[0,0,-0.1,-1000], 
[0,0.1,-0.2,-1000], 
[0,0.1,-0.2,-1000], 
[0,0,0.6,-1000], 
[0,-1000,1.1,-1000]
])

# Coordinates for visualization
co_ordinate = np.array([
[11.47831871,6.103130316], 
[12.55703574,3.364647586], 
[12.99208075,0.453693457], 
[12.76115338,-2.48051694], 
[11.87609095,-5.28757636], 
[10.38226163,-7.823595301], 
[8.356238926,-9.958577761], 
[12.35510718,3.322312561], 
[12.78580429,0.457867519], 
[12.56110274,-2.430047752], 
[11.69252069,-5.193399091], 
[10.22458154,-7.690537465], 
[8.232531664,-9.793459885], 
[5.818483284,-11.39437082], 
[12.17710377,0.411982649], 
[11.95768132,-2.337828772], 
[11.12530968,-4.967803394], 
[9.722656102,-7.343129057], 
[7.821620433,-9.342046877], 
[5.519649513,-10.86209258], 
[2.934742075,-11.82534883], 
[10.96795432,-2.241253677], 
[10.18267398,-4.651063378], 
[8.875431097,-6.822460174], 
[7.113234764,-8.644138544], 
[4.986414938,-10.02271949], 
[2.603992129,-10.88753714], 
[0.088089021,-11.19426104], 
[8.888281192,-4.284942767], 
[7.696572728,-6.174548186], 
[6.110338943,-7.747647066], 
[4.210889975,-8.923602562], 
[2.095591333,-9.642135346], 
[-0.127127049,-9.866413521], 
[-2.343328915,-9.584940618], 
[6.214346399,-5.447602704], 
[4.829629131,-6.705904774], 
[3.1973457,-7.620463035], 
[1.401166743,-8.144397347], 
[-0.46683584,-8.250850907], 
[-2.310908479,-7.934366918], 
[-4.036524248,-7.21116831], 
[3.301353157,-5.574786736], 
[1.962685535,-6.174548186], 
[0.523410907,-6.457803097], 
[-0.942693696,-6.410031857], 
[-2.360475942,-6.033683215], 
[-3.657260497,-5.348048754], 
[-4.766574354,-4.388274006]
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
