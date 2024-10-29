# Crawling Robot with Machine Learning Control

## Project Overview

This project explores the application of Machine Learning (ML) in controlling a simple mechatronic system—a crawling robot powered by a single servo-driven arm. It draws comparisons between ML and Direct Programming (DP) approaches to examine the benefits and limitations of using ML in robotic movement control.

The ML approach utilizes Q-learning, a reinforcement learning algorithm, to enable the robot to learn optimal movements autonomously through iterative trial and error. The goal is to maximize the robot's movement efficiency by adapting its actions based on reward feedback. This ML-based movement control is then evaluated against a traditional DP approach to compare performance metrics, specifically average velocity, development time, and skill requirements.

### Key Findings
- **Performance**: The ML-controlled crawling robot achieved a higher average velocity compared to the DP approach, demonstrating the potential for ML to enhance movement efficiency in robotic applications.
- **Development Time & Skill Requirement**: Implementing ML required significantly more time and a specialized skillset, including knowledge of reinforcement learning and reward systems, compared to the straightforward DP method.
- **Conclusion**: While ML provides performance benefits, it may not always be the best choice due to the increased development time and expertise needed.

## Project Structure

### Code and Simulation

The project consists of Python scripts for Q-learning-based ML training and testing on simulated states. Each state represents a possible configuration of the crawling robot's servo arm. The scripts train the robot to take actions (movements) that maximize cumulative rewards, with physical testing conducted on a Raspberry Pi.

### Code Features

1. **Q-Learning Algorithm**: Implemented to allow the robot to learn optimal actions across 49 states, each representing a position and movement possibility.
2. **Epsilon-Greedy Strategy**: An epsilon decay function controls the exploration-exploitation balance, allowing the robot to explore new movements while focusing on high-reward actions as training progresses.
3. **Transition and Reward Matrices**: Define the allowable actions and associated rewards, which guide the robot in finding the most efficient crawling movements.
4. **Path Visualization**: The trained model's actions are plotted to visualize the robot’s learned path and evaluate the efficiency of its movements.

### Hardware Setup

The ML-controlled crawling robot uses a Raspberry Pi and a single servo motor to achieve movement. The project includes:
- **Motor Control**: GPIO pins on the Raspberry Pi control motor actions, allowing for precise control during training and testing.
- **Rotary Encoder**: Used to capture the robot’s displacement, essential for measuring the reward and evaluating performance.
- **Temperature Sensor**: Monitors motor conditions, protecting hardware from overheating during extended training sessions.

### Requirements

- **Hardware**:
  - Raspberry Pi with GPIO control
  - Servo motor and rotary encoder
  - Temperature sensor (optional for motor protection)
- **Software**:
  - Python 3.x
  - Libraries: `numpy`, `matplotlib`, `RPi.GPIO`, `xlsxwriter`

## Usage

### Train the Robot
Run the script to start Q-learning training. Adjust parameters such as `loops1`, `loops2`, `epsilon_value`, `learning_rate`, and `discount_factor` as needed.

### Evaluate Performance
After training, initiate the evaluation phase to test the robot's learned movements. The path and actions taken will be plotted and saved, allowing for performance analysis.

### Generate Reports
The code outputs performance metrics to Excel, enabling easy comparison of ML and DP performance metrics.

## Results and Discussion
The Q-learning-based approach allowed the robot to learn and adapt its movement strategy, producing a higher average velocity than DP control. However, this improvement came at the cost of a more complex implementation, longer development time, and the need for a foundational knowledge of ML concepts. Thus, while ML control demonstrates benefits in movement efficiency, DP may be preferred for simpler, faster-to-develop projects.

## Future Work
Potential expansions for this project include:

- **Enhanced Reward System**: Introducing additional factors, such as power consumption or obstacle navigation, could refine the training process.
- **Real-Time Adaptation**: Implementing online learning to allow the robot to adapt to new environments dynamically.
- **Improved Hardware**: Using more advanced sensors and motors could increase the robot’s movement precision, enhancing the learning algorithm’s effectiveness.

## Conclusion
This project demonstrates that while ML can enhance the efficiency of simple mechatronic systems, it may not always be the most feasible choice. Trade-offs between performance improvements and development time must be considered, especially in applications where quick deployment or limited skillsets are factors.
