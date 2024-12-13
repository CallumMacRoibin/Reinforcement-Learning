import numpy as np
import random
import matplotlib.pyplot as plt

# Hyperparameters
start_state = 21
loops1 = 300
loops2 = 30
discount_factor = 0.95
learning_rate = 0.15
epsilon = 1     # Start value

# Initialize Q-matrix
q_matrix = np.zeros((49, 4))

# Define valid actions and transitions
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

reward = np.array([
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

# Training phase
for episode in range(loops1):
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
current_state = start_state
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
        current_state += 7
    elif step == 3:
        print("go down")
        current_state -= 7
    actions.append(int(step))
    states.append(current_state)

print("Actions Taken:", actions)
print("States:", states)
