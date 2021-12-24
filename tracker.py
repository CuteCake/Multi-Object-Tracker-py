#Test a multi-object tracker on the environment from environment.py
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import pygame
from enviroment import Point, PointsEnv
from motionModel import ConstantVelocityModel


class BaseTracker: #This is a simple Extended Kalman Filter tracker
    '''
    This is a simple Extended Kalman Filter tracker
    It uses a prediction model, and update the state space for the prediction step
    It then generates a Kalman Gain
    And uses the Kalman Gain to correct the prediction
    '''
    def __init__(self, sensor_noise = 5, measurement_noise = 5, 
        state_noise = 5, motion_model = 'constant velocity'):
        self.sensor_noise = sensor_noise
        self.measurement_noise = measurement_noise
        self.state_noise = state_noise

        #The following are for logging purposes
        self.state_estimate = []
        self.state_estimate_history = []

        self.measurement_history = []
        self.prediction_history = []


        if motion_model == 'constant velocity':
            self.motionFilter = ConstantVelocityModel()
        elif motion_model == 'constant turning rate':
            raise Exception('Measurement model not supported')
            self.motionFilter = self.ConstantTurningRateMeasurementModel
        else:
            raise Exception('Measurement model not supported')

    def updateTracker(self, observation, dt):
        self.state_estimate = self.motionFilter.update(observation, dt)

        return self.state_estimate



if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    env = PointsEnv(640, 480, 10)
    tracker = BaseTracker()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(env.get_time_elapsed())
                pygame.quit()
                quit()
        screen.fill((0, 0, 0))
        env.draw(screen)
        observation = env.update()
        
        env.draw_observed_points(screen, observation)
        obs = np.array(observation[0])
        stateVec = tracker.updateTracker(obs, env.get_last_dt())
        # env.draw_prediction(screen, stateVec)
        pygame.draw.circle(screen, (10,10, 255), (int(stateVec[0]),int(stateVec[1])), 5)
        pygame.display.update()
