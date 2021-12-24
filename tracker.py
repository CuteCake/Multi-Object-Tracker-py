#Test a multi-object tracker on the environment from environment.py
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import pygame
from enviroment import Point, PointsEnv
from motionModel import ConstantVelocityFilter, ConstantVelocityConstantTurningRateFilter

class BaseTracker: #Base class for tracker, should be inherited (and overwritten by real tracker)
    def __init__(self) -> None:
        pass
    def updateTracker(self, observation, dt) -> None:
        raise NotImplementedError
        return self.state_estimate  
        
class SingleTracker(BaseTracker): #This is a simple Extended Kalman Filter tracker
    '''
    This is a single object Kalman Filter tracker
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
            self.motionFilter = ConstantVelocityFilter()
        elif motion_model == 'constant turning rate':
            self.motionFilter = ConstantVelocityConstantTurningRateFilter()
        else:
            raise Exception('Measurement model not supported')

    def updateTracker(self, observation, dt):
        print('dt: ', dt)
        self.state_estimate = self.motionFilter.update(observation, dt)

        return self.state_estimate

class MultiTracker(BaseTracker): #This is a multi-object Kalman Filter tracker
    '''
    This is a multi-object Kalman Filter tracker
    observations -> [[x,y],[x,y],...]

    1. do data association (between detected objects with tracked objects)
        Method 1: GNN Global Nearest Neighbor
            It calculates the Mahalanobis distance (considers both position and covarience)
             between each detected object and tracked objects
        Method 2: JPDA Joint Probability Distance Association

    2. do the track management (delete, create tracks)
        Create: 
            1. When a new detection is there, create a new tentative track in the
            background and associate it with the detection
            2. When a tentative track is associated with a detection for a continuous
            of some times, put the track into the confirmed tracks

        Delete: When a detection is not detected for a continuous of some time
                , delete the track
        
    3. update the Kalman Filter for each object
        Method 1: Use a single filter (KF,EKF,UKF, on some motion model)
        Method 2: Use a IMM Interactive Multi-Object Kalman Filter
              It updates one object using several models then outputs a weighted sum

    4. do gating:
        This is mainly for JPDA
        if the measurement is not in the gate range, don't waste the computation

    5. retun the state_estimate
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
            self.motionFilter = ConstantVelocityFilter()
        elif motion_model == 'constant turning rate':
            self.motionFilter = ConstantVelocityConstantTurningRateFilter()
        else:
            raise Exception('Measurement model not supported')
    def dataAssociation(self, observations, state_estimate):
        pass

    def trackManagement(self, observations, state_estimate):
        pass

    def updateTracker(self, observation, dt):

        return self.state_estimate

    def gating(self, observations, state_estimate):
        pass


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    env = PointsEnv(640, 480, 10)
    tracker = SingleTracker()
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
