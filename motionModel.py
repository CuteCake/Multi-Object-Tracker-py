import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from enviroment import Point, PointsEnv

class ConstantVelocityModel:
    '''
    The state space is defined as:
    [x, y, vx, vy]: x,y are the position, vx,vy are the velocities

    The observation space is defined as:
    [x, y]

    The 
    '''
    def __init__(self, x=0, y=0, vx=0, vy=0, \
        stateNoise=1,observationNoise=10, id=None):
        #These are the state variables:
        #This method is not as efficient as using a numpy array,
        #  but it is easier to read
        #However, whenever we output, we output as list, so we can use numpy array

        self.stateVector = np.array([x, y, vx, vy]).T #It is a column vector
        self.stateTransitionCovariance = \
            np.array(   [[1, 0, 0, 0], \
                        [0, 1, 0, 0], \
                        [0, 0, 1, 0], \
                        [0, 0, 0, 1], ]) * stateNoise
        self.observationCovariance = \
            np.array(   [[1, 0], \
                        [0, 1], ]) * observationNoise
        self.observationMatrix = np.array([[1, 0, 0, 0], \
                                            [0, 1, 0, 0]])
        self.stateCovariance = np.eye(4) * stateNoise
        self.observationCovariance = np.eye(2) * observationNoise
        self.id = id

    def getStateUpdateMatrix(self, dt): #Get state estimation but don't update
        self.stateUpdateMatrix = np.array([[1, 0, dt, 0], \
                                             [0, 1, 0, dt], \
                                                [0, 0, 1, 0], \
                                                [0, 0, 0, 1]])
        return self.stateUpdateMatrix

    def setState(self, state):
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]
        self.v = state[3]

    def update(self, observation, dt):
        '''
        observation: [x,y]
        dt: time since last update
        '''
        #Prediction step
        stateUpdateMatrix = self.getStateUpdateMatrix(dt)
        stateE = stateUpdateMatrix.dot(self.stateVector)
        stateCovarianceE = stateUpdateMatrix.dot(self.stateCovariance).dot(stateUpdateMatrix.T) + \
            self.stateTransitionCovariance
        #Generate Kalman Gain
        kalmanGain = stateCovarianceE.dot(self.observationMatrix.T).dot(np.linalg.inv(self.observationCovariance + \
            self.observationMatrix.dot(stateCovarianceE).dot(self.observationMatrix.T)))
        #Correct prediction
        # kalmanGain = kalmanGain[::2]
        print(kalmanGain)
        self.stateVector = stateE + kalmanGain.dot(np.array(observation).T - self.observationMatrix.dot(stateE))
        self.stateCovariance = (np.eye(4) - kalmanGain.dot(self.observationMatrix)).dot(stateCovarianceE)
        return self.stateVector