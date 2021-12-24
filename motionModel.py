import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from enviroment import Point, PointsEnv

class BaseFilter: #This is a template for motion filters, should be overwritten
    def __init__(self):
        pass
    def update(self, observation, dt):
        raise NotImplementedError
        return observation

class ConstantVelocityFilter(BaseFilter): 
    '''
    Constent velocity model Kalman Filter, not EKF!
    The state space is defined as:
    [x, y, vx, vy]: x,y are the position, vx,vy are the velocities

    The observation space is defined as:
    [x, y]

    The 
    '''
    def __init__(self, x=0, y=0, vx=0, vy=0, \
        stateNoise=0.5,observationNoise=10, id=None):
        #state variables in Numpy array
        #[x, y, vx, vy].T

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
        self.stateVector = stateE + kalmanGain.dot(np.array(observation).T - self.observationMatrix.dot(stateE))
        self.stateCovariance = (np.eye(4) - kalmanGain.dot(self.observationMatrix)).dot(stateCovarianceE)
        return self.stateVector

    # def setState(self, state):
    #     self.x = state[0]
    #     self.y = state[1]
    #     self.theta = state[2]
    #     self.v = state[3]

    class ConstantVelocityConstantTurningRateFilter(BaseFilter):
        def __init__(self, x=0, y=0, v=0, twist=0, turnRate=0, \
            stateNoise=0.5,observationNoise=10, id=None):
            #These are the state variables:
            self.stateVector = np.array([x, y, twist, v, turnRate]).T

        def update(self, observation, dt):
            return super().update(observation, dt)
