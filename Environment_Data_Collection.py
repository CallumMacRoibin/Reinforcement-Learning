# Enviroment Data
# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO
import time
from time import sleep
import sys
import math
import datetime
import xlsxwriter

# Initial setup
starting_state = 0

# Define physical system variables (GPIO pins and parameters)
m1 = 21                   # Pin for Servo Motor 1
m2 = 20                   # Pin for Servo Motor 2
clk = 14                  # Pin for Rotary Encoder (Clock)
dt = 15                   # Pin for Rotary Encoder (Data)

# Servo position increment degrees
ang_increment_m1 = 13     # Degrees for motor 1 increment
ang_increment_m2 = 21     # Degrees for motor 2 increment

# Initial servo positions
m1_start = 8
m2_start = 4.277777778

# Initialize matrices and arrays
change_matrix = np.zeros((49, 4))  # Matrix to store changes
state_duty = np.array([           # State matrix for motor duty cycles
    [8, 4.277777778], [7.277777778, 4.277777778], [6.555555556, 4.277777778],
    [5.833333333, 4.277777778], [5.111111111, 4.277777778], [4.388888889, 4.277777778],
    # More states follow here...
    [3.666666667, 11.27777778]    # Final state
])

# Valid actions matrix
valid_actions = np.array([
    [-1, 1, -1, 3], [0, 1, -1, 3], [0, 1, -1, 3],
    # Complete the matrix here
    [0, -1, 2, -1]  # Final state actions
])

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(m1, GPIO.OUT)
GPIO.setup(m2, GPIO.OUT)
GPIO.setup(clk, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(dt, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Initialize PWM for motors
mAG = GPIO.PWM(m1, 50)  # Motor 1 PWM
m17 = GPIO.PWM(m2, 50)  # Motor 2 PWM

# Calculate duty cycle changes and delays for motors
m1_duty_change = ang_increment_m1 * 0.055555556
m2_duty_change = ang_increment_m2 * 0.055555556
delay_m1 = (ang_increment_m1 * (0.1 / 60)) + 0.1
delay_m2 = (ang_increment_m1 * (0.1 / 60)) + 0.1

# Rotary encoder setup
clkLastState = GPIO.input(clk)
counter = 0  # Initialize counter

# Start motor PWM at initial positions
mAG.start(m1_start)
m17.start(m2_start)
sleep(2)

# Rotary Encoder Callback
def rotary_encoder_callback(clk):
    global clkLastState, counter
    clkState = GPIO.input(clk)
    dtState = GPIO.input(dt)
    
    if clkState != clkLastState:
        counter += 1 if dtState != clkState else -1
    clkLastState = clkState

GPIO.add_event_detect(clk, GPIO.BOTH, callback=rotary_encoder_callback)

# Main loop for updating state and duty cycles
for i in range(49):
    state = i
    print(f"State number {state}")
    state_m1, state_m2 = state_duty[state]

    for action in range(4):
        x = action
        sleep(1)
        mAG.ChangeDutyCycle(state_m1)
        m17.ChangeDutyCycle(state_m2)
        sleep(1)

        # Check if action is valid
        if valid_actions[state][x] != x:
            change_matrix[state][x] = -1000
        else:
            # Perform action
            old_disp = counter
            if action == 0:  # Move Motor 1 forward
                m1_action = state_m1 + m1_duty_change
                mAG.ChangeDutyCycle(m1_action)
                sleep(delay_m1)
            elif action == 1:  # Move Motor 1 backward
                m1_action = state_m1 - m1_duty_change
                mAG.ChangeDutyCycle(m1_action)
                sleep(delay_m1)
            elif action == 2:  # Move Motor 2 backward
                m2_action = state_m2 - m2_duty_change
                m17.ChangeDutyCycle(m2_action)
                sleep(delay_m2)
            elif action == 3:  # Move Motor 2 forward
                m2_action = state_m2 + m2_duty_change
                m17.ChangeDutyCycle(m2_action)
                sleep(delay_m2)
                
            # Record the change
            change = counter - old_disp
            change_matrix[state][x] = change
            print(f"Action {action}")

    print(change_matrix[state])

# Save change matrix to Excel
time_now = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
workbook = xlsxwriter.Workbook(f"Change_Matrix_{time_now}.xlsx")
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "Change_Matrix")

# Write change matrix to file
row = 1
for action_0, action_1, action_2, action_3 in change_matrix:
    worksheet.write(row, 0, action_0)
    worksheet.write(row, 1, action_1)
    worksheet.write(row, 2, action_2)
    worksheet.write(row, 3, action_3)
    row += 1

workbook.close()
sys.exit()
