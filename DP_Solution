# Setup
# Import Modules
import RPi.GPIO as GPIO
import time
from time import sleep
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import xlsxwriter

print("Code Running")

# Setup variables
m1 = 21                # Pin of servo motor 1
m2 = 20                # Pin of servo motor 2
clk = 14               # Pin of clk on rotary
dt = 15                # Pin of dt on rotary
ang_increment_m1 = 13  # Angle Between M1 Positions
ang_increment_m2 = 21  # Angle Between M2 Positions
resolution = 3.75      # Rotary Encoder Resolution (mm)
clicks = 270           # Rotary Encoder Positions Desired

# Motor 1 Servo Duty Cycle Positions
a = 8
b = 7.277777778
c = 6.555555556
d = 5.833333333
e = 5.111111111
f = 4.388888889
g = 3.666666667

# Motor 2 Servo Duty Cycle Positions
aa = 4.277777778
bb = 5.444444444
cc = 6.611111111
dd = 7.777777778
ee = 8.944444444
ff = 10.11111111
gg = 11.27777778

# Motor Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(m1, GPIO.OUT)
GPIO.setup(m2, GPIO.OUT)
GPIO.setup(clk, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(dt, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
mAG = GPIO.PWM(m1, 50) # m1 - positions A - G
m17 = GPIO.PWM(m2, 50) # m2 - positions 1 - 7
delay_m1 = ang_increment_m1 * (0.1 / 60) + 0.1  # Delay between M1 movements
delay_m2 = ang_increment_m2 * (0.1 / 60) + 0.1  # Delay between M2 movements

# Rotary Setup
clkLastState = GPIO.input(clk)
counter = 0
vel_graph_disp = []
vel_graph_time = []
time_disp_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

# Rotary Encoder Function
def rotary_encoder_callback(clk):
    global clkLastState, counter
    clkState = GPIO.input(clk)
    dtState = GPIO.input(dt)
    if clkState != clkLastState:
        if dtState != clkState:
            counter += 1
        else:
            counter -= 1
        disp = counter * resolution
        vel_graph_disp.append(disp)
        time_disp_current = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        time_disp = (time_disp_current - time_disp_start) / 1_000_000_000
        vel_graph_time.append(time_disp)
        clkLastState = clkState

GPIO.add_event_detect(clk, GPIO.BOTH, callback=rotary_encoder_callback)

# Loop Preparation
mAG.start(a)     # Start Motors in Position A
m17.start(dd)
sleep(2)
time1 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
print("Code Initialized")
print("Test Starting")

while counter < clicks:
    if counter < clicks:
        m17.ChangeDutyCycle(cc)
        sleep(delay_m2)
    else:
        break

    if counter < clicks:
        mAG.ChangeDutyCycle(b)
        sleep(delay_m1)
    else:
        break

    if counter < clicks:
        m17.ChangeDutyCycle(bb)
        sleep(delay_m2)
    else:
        break

    if counter < clicks:
        mAG.ChangeDutyCycle(c)
        sleep(delay_m1)
    else:
        break

    if counter < clicks:
        m17.ChangeDutyCycle(aa)
        sleep(delay_m2)
    else:
        break

    if counter < clicks:
        mAG.ChangeDutyCycle(d)
        sleep(delay_m1)
    else:
        break

    if counter < clicks:
        m17.ChangeDutyCycle(bb)
        sleep(delay_m2)
    else:
        break

    if counter < clicks:
        mAG.ChangeDutyCycle(e)
        sleep(delay_m1)
    else:
        break

    if counter < clicks:
        m17.ChangeDutyCycle(cc)
        sleep(delay_m2)
    else:
        break

    if counter < clicks:
        m17.ChangeDutyCycle(dd)
        sleep(delay_m2)
    else:
        break

    if counter < clicks:
        m17.ChangeDutyCycle(ee)
        sleep(delay_m2)
    else:
        break

    if counter < clicks:
        m17.ChangeDutyCycle(ff)
        sleep(delay_m2)
    else:
        break

    if counter < clicks:
        m17.ChangeDutyCycle(gg)
        sleep(delay_m2)
    else:
        break

    if counter < clicks:
        mAG.ChangeDutyCycle(d)
        sleep(delay_m1)
    else:
        break

    if counter < clicks:
        mAG.ChangeDutyCycle(c)
        sleep(delay_m1)
    else:
        break

    if counter < clicks:
        mAG.ChangeDutyCycle(b)
        sleep(delay_m1)
    else:
        break

    if counter < clicks:
        m17.ChangeDutyCycle(ff)
        sleep(delay_m2)
    else:
        break

    if counter < clicks:
        m17.ChangeDutyCycle(ee)
        sleep(delay_m2)
    else:
        break

    if counter < clicks:
        mAG.ChangeDutyCycle(a)
        sleep(delay_m1)
    else:
        break

    if counter < clicks:
        m17.ChangeDutyCycle(dd)
        sleep(delay_m2)
    else:
        break

# Evaluation
time2 = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
timetotal = (time2 - time1) / 1_000_000_000
time_now = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
displacement = counter * resolution
velocity = displacement / timetotal
print("Test Completed")

# Velocity Graph
plt.plot(vel_graph_time, vel_graph_disp)
plt.xlabel("Time (s)")
plt.ylabel("Displacement (mm)")
plt.title("Velocity Profile")
plt.savefig('DP_Figs/Velocity_Graph_' + time_now + '.png')

# Export Data to Excel
workbook = xlsxwriter.Workbook('DP_XL/DP_Velocity_' + time_now + ".xlsx")
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "Displacement")
for row, displacement in enumerate(vel_graph_disp, start=1):
    worksheet.write(row, 0, displacement)

worksheet.write(0, 2, "Time")
for row, seconds in enumerate(vel_graph_time, start=1):
    worksheet.write(row, 2, seconds)

workbook.close()

print(f"The time taken to complete {displacement} mm was {timetotal:.2f} seconds")
print(f'The average velocity was {velocity:.2f} mm/s')
print("Code finished")
sys.exit()
