import numpy as np
import random
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO
import time
from time import sleep
import sys
import math
import datetime
import xlsxwriter

# Setup Variables
## Machine Learning Variables
starting_state = 21
learn_rate = 0.1
discount_factor = 0.94
epsilon_value = 1
loops = 1000  # Total iterations
loops2 = 30   # Actions per iteration

## Physical System Variables
m1 = 21                  # Pin of servo motor 1
m2 = 20                  # Pin of servo motor 2
clk = 14                 # Pin of clk on rotary encoder
dt = 15                  # Pin of dt on rotary encoder
temp_sensor = 18         # Pin of temperature sensor
ang_increment_m1 = 13    # Degree swing for motor 1
ang_increment_m2 = 21    # Degree swing for motor 2
resolution = 3.75        # Rotary encoder resolution in mm per click
clicks = 270             # Target clicks (40 clicks = 1 revolution)
m1_start = 8             # Starting duty cycle for motor 1
m2_start = 7.778         # Starting duty cycle for motor 2
motor_overheat = 0       # Monitor motor temperature

## Q-Learning Setup
q_matrix = np.zeros((49, 4))  # Initialize Q-matrix with all zeros
# Action index mapping: 0 = left, 1 = right, 2 = up, 3 = down
valid_actions = np.array([
    [1, 3], [0, 1, 3], [0, 1, 3], [0, 1, 3], [0, 1, 3],
    [0, 1, 3], [0, 3], [1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
    [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3],
    [1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
    [0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3], [1, 2, 3],
    [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
    [0, 1, 2, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3],
    [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
    [0, 1, 2, 3], [0, 2, 3], [1, 2], [0, 1, 2], [0, 1, 2],
    [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 2]
])

# Transition matrix for state-action transitions
transition_matrix = np.array([
    [-1, 1, -1, 7], [0, 2, -1, 8], [1, 3, -1, 9], [2, 4, -1, 10], [3, 5, -1, 11],
    [4, 6, -1, 12], [5, -1, -1, 13], [-1, 8, 0, 14], [7, 9, 1, 15], [8, 10, 2, 16],
    [9, 11, 3, 17], [10, 12, 4, 18], [11, 13, 5, 19], [12, -1, 6, 20], [-1, 15, 7, 21],
    [14, 16, 8, 22], [15, 17, 9, 23], [16, 18, 10, 24], [17, 19, 11, 25], [18, 20, 12, 26],
    [19, -1, 13, 27], [-1, 22, 14, 28], [21, 23, 15, 29], [22, 24, 16, 30], [23, 25, 17, 31],
    [24, 26, 18, 32], [25, 27, 19, 33], [26, -1, 20, 34], [-1, 29, 21, 35], [28, 30, 22, 36],
    [29, 31, 23, 37], [30, 32, 24, 38], [31, 33, 25, 39], [32, 34, 26, 40], [33, -1, 27, 41],
    [-1, 36, 28, 42], [35, 37, 29, 43], [36, 38, 30, 44], [37, 39, 31, 45], [38, 40, 32, 46],
    [39, 41, 33, 47], [40, -1, 34, 48], [-1, 43, 35, -1], [42, 44, 36, -1], [43, 45, 37, -1],
    [44, 46, 38, -1], [45, 47, 39, -1], [46, 48, 40, -1], [47, -1, 41, -1]
])

# Motor Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(m1, GPIO.OUT)
GPIO.setup(m2, GPIO.OUT)
GPIO.setup(clk, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(dt, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
mAG = GPIO.PWM(m1, 50)  # Motor 1 at 50 Hz
m17 = GPIO.PWM(m2, 50)  # Motor 2 at 50 Hz

# Duty Cycle Changes for Motors
m1_duty_change = ang_increment_m1 * 0.055555556
m2_duty_change = ang_increment_m2 * 0.055555556

# Delay between motor movements
delay_m1 = (ang_increment_m1 * (0.1 / 60)) + 0.1
delay_m2 = (ang_increment_m2 * (0.1 / 60)) + 0.1

# Rotary Encoder Setup
clkLastState = GPIO.input(clk)
counter = 0  # Start counter at 0
vel_graph_disp = []
vel_graph_time = []
time_disp_start = 0

# Rotary Encoder Callback Function
def rotary_encoder_callback(clk):
    global clkLastState, counter, disp, time_disp_current
    clkState = GPIO.input(clk)
    dtState = GPIO.input(dt)
    if clkState != clkLastState:
        counter += 1 if dtState != clkState else -1
        disp = counter * resolution
        time_disp_current = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    clkLastState = clkState

GPIO.add_event_detect(clk, GPIO.BOTH, callback=rotary_encoder_callback)

# Epsilon Exponential Decay Function
A, B, C = 0.9, 0.15, 0.1
def epsilon(itt):
    exp_value = (itt - A * loops) / (B * loops)
    cosh = np.cosh(math.exp(-exp_value))
    return 1 - (1 / cosh + (itt * C / loops))

# Temperature Sensor Setup
GPIO.setup(temp_sensor, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Loop Preparation
mAG.start(m1_start)  # Start Motor 1 in Position A4
m17.start(m2_start)  # Start Motor 2 in Position 7.778
sleep(2)

print("Training beginning")
for i in range(loops):
    try:
        loop_number = i
        current_state = starting_state
        print(f"Iteration {i}, Epsilon = {epsilon_value:.2f}")

        # Initialize Motor Positions
        m1_action, m2_action = m1_start, m2_start
        mAG.ChangeDutyCycle(m1_action)
        m17.ChangeDutyCycle(m2_action)
        sleep(0.5)

        # Epsilon-Greedy Strategy for Exploration and Exploitation
        if random.uniform(0, 1) < epsilon_value:
            print("Exploration")
            action = random.choice(valid_actions[current_state])
        else:
            print("Exploitation")
            q_values = q_matrix[current_state, :]
            action = np.argmax(q_values)

        # Execute Action
        if GPIO.input(temp_sensor) == GPIO.LOW:
            if action == 0:
                m1_action += m1_duty_change
                mAG.ChangeDutyCycle(m1_action)
                sleep(delay_m1)
            elif action == 1:
                m1_action -= m1_duty_change
                mAG.ChangeDutyCycle(m1_action)
                sleep(delay_m1)
            elif action == 2:
                m2_action -= m2_duty_change
                m17.ChangeDutyCycle(m2_action)
                sleep(delay_m2)
            elif action == 3:
                m2_action += m2_duty_change
                m17.ChangeDutyCycle(m2_action)
                sleep(delay_m2)
        else:
            # Motor Cooling Loop
            print("Motor Overheat")
            while GPIO.input(temp_sensor) == GPIO.HIGH:
                print("Motor Cooling")
                sleep(5)

        # Reward System
        change = counter - old_disp
        reward = change if change != 0 else 0.3

        # Q-Value Update Using Bellman Equation
        next_state = transition_matrix[current_state][action]
        future_rewards = [q_matrix[next_state][a] for a in valid_actions[next_state]]
        old_q = q_matrix[current_state][action]
        new_q = reward + (discount_factor * max(future_rewards))
        q_matrix[current_state][action] = old_q + learn_rate * (new_q - old_q)
        current_state = next_state

        # Update Epsilon Value
        if epsilon_value > 0 and i - 1 == loops2:
            epsilon_value = epsilon(loop_number)
        
    except KeyboardInterrupt:
        # Pause and Resume Functionality
        print("\nPausing... Hit ENTER to continue")
        mAG.ChangeDutyCycle(m1_action)
        m17.ChangeDutyCycle(m2_action)
        try:
            if input("Type 'quit' to exit") == "quit":
                break
        except KeyboardInterrupt:
            print("Resuming...")
            continue

print("Training Over")

# Evaluation
## Reset Variables
current_state = starting_state
m1_action = m1_start
m2_action = m2_start
mAG.ChangeDutyCycle(m1_action)
m17.ChangeDutyCycle(m2_action)
counter = 0
disp = 0
test = 0
actions = []

# Wait for User Input to Start Evaluation
while test == 0:
    test1 = input("Please type 'y' to begin performance evaluation: ")
    if test1.lower() == "y":
        test = 1

sleep(2)
print("Evaluation Beginning")

# Test Policy
time_disp_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
while counter < clicks:
    # Select Action with Highest Q-Value
    q_values = q_matrix[current_state, :]
    action = np.argmax(q_values) if q_values.any() else random.choice(valid_actions[current_state])
    actions.append(action)
    next_state = transition_matrix[current_state][action]

    # Execute Action
    if action == 0 and counter < clicks:
        m1_action += m1_duty_change
        mAG.ChangeDutyCycle(m1_action)
        sleep(delay_m1)
    elif action == 1 and counter < clicks:
        m1_action -= m1_duty_change
        mAG.ChangeDutyCycle(m1_action)
        sleep(delay_m1)
    elif action == 2 and counter < clicks:
        m2_action -= m2_duty_change
        m17.ChangeDutyCycle(m2_action)
        sleep(delay_m2)
    elif action == 3 and counter < clicks:
        m2_action += m2_duty_change
        m17.ChangeDutyCycle(m2_action)
        sleep(delay_m2)
    else:
        break  # Exit if counter has reached 'clicks' threshold

    # Append Data
    vel_graph_disp.append(disp)
    time_disp_current = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    time_disp = (time_disp_current - time_disp_start) / 1_000_000_000
    vel_graph_time.append(time_disp)
    current_state = next_state

# Calculate Results if Evaluation Completes
if counter >= clicks:
    time2 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    timetotal = (time2 - time_disp_start) / 1_000_000_000
    time_now = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    displacement = counter * resolution
    velocity = displacement / timetotal

    # Generate and Save Velocity Graph
    plt.plot(vel_graph_time, vel_graph_disp)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (mm)")
    plt.title("Velocity Profile")
    plt.savefig(f'RL_Figs/Velocity_Profile_{time_now}.png')

    # Export Data to Excel
    workbook = xlsxwriter.Workbook(f'RL_XL/RL_Code_{time_now}.xlsx')
    worksheet = workbook.add_worksheet()

    # Write Q-Matrix to Excel
    worksheet.write(0, 0, "Q-Matrix")
    for row, (action_0, action_1, action_2, action_3) in enumerate(q_matrix, start=1):
        worksheet.write_row(row, 0, [action_0, action_1, action_2, action_3])

    # Write Displacement Data to Excel
    worksheet.write(0, 5, "Displacement")
    for row, displacement in enumerate(vel_graph_disp, start=1):
        worksheet.write(row, 5, displacement)

    # Write Time Data to Excel
    worksheet.write(0, 7, "Time")
    for row, seconds in enumerate(vel_graph_time, start=1):
        worksheet.write(row, 7, seconds)

    # Write Actions Data to Excel
    worksheet.write(0, 9, "Actions")
    for row, action in enumerate(actions, start=1):
        worksheet.write(row, 9, action)

    workbook.close()
    print(f"The time taken to complete {displacement} mm was {timetotal:.2f} seconds")
    print(f"The average velocity was {velocity:.2f} mm/s")
    print("Code finished")


print ("The time taken to complete %d mm was %d seconds" %(displacement, timetotal)) 
print ('The average velocity was %d mm/s' %velocity) 
print ("Code finished") 
sys.exit()
