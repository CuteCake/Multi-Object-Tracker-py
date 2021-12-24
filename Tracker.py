#Test a multi-object tracker on the environment from environment.py
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import pygame
from enviroment import Point, PointsEnv

class StateSpaceConstantSpeed:
    def __init__(self, x, y, theta, v,id):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.id = id

    def update(self, dt):
        self.x += self.v*dt*np.cos(self.theta)
        self.y += self.v*dt*math.sin(self.theta)
        self.theta += self.v*dt*math.tan(self.theta)
        self.theta = self.theta%(2*math.pi)

class BaseTracker: #This is a simple Extended Kalman Filter tracker
    '''
    This is a simple Extended Kalman Filter tracker
    It uses a constant velocity model, this update is for the prediction step
    It then generates a Kalman Gain
    And uses the Kalman Gain to correct the prediction
    '''
    def __init__(self, sensor_noise = 5, measurement_noise = 5, state_noise = 5, measurement_model = 'constant velocity'):
        self.sensor_noise = sensor_noise
        self.measurement_noise = measurement_noise
        self.state_noise = state_noise
        self.measurement_model = measurement_model
        self.state_estimate = []
        self.state_estimate_history = []
        self.measurement_estimate = []
        self.measurement_estimate_history = []
        self.prediction_estimate = []
        self.prediction_estimate_history = []
        self.kalman_gain = []
        self.kalman_gain_history = []
        self.measurement_history = []
        self.prediction_history = []
        self.measurement_model = measurement_model
        self.sensor_noise = sensor_noise
        self.measurement_noise = measurement_noise
        self.state_noise = state_noise
        self.state_estimate = []
        self.state_estimate_history = []
        self.measurement_estimate = []
        self.measurement_estimate_history = []
        self.prediction_estimate = []
        self.prediction_estimate_history = []
        self.kalman_gain = []
        self.kalman_gain_history = []
        self.measurement_history = []
        self.prediction_history = []
        

    def update(self, observation, dt):
        if self.measurement_model == 'constant velocity':
            #Prediction step
            self.prediction_estimate = None
            #Generate Kalman Gain
            self.kalman_gain = self.generate_kalman_gain(self.prediction_estimate, observation)
            #Correct prediction
            self.state_estimate = self.correct(self.prediction_estimate, self.kalman_gain, observation)


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    env = PointsEnv(640, 480, 10)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(env.get_time_elapsed())
                pygame.quit()
                quit()
        screen.fill((0, 0, 0))
        env.draw(screen)
        env.update(0.03)
        observation = env.update(dt=0.03)
        env.draw_observed_points(screen, observation)
        pygame.display.update()