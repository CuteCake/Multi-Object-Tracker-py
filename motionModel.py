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

    The state transition model is:
    x(k+1) = x(k) + vx(k) * dt
    y(k+1) = y(k) + vy(k) * dt
    vx(k+1) = vx(k)
    vy(k+1) = vy(k)
    It is linear, that's why it can be represented as a matrix.
    But in CVCT it can only be represented as a bunch of equations.

    The observation model is:
    x(k) = x(k)
    y(k) = y(k)
    It is linear, that's why it can be represented as a matrix.
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
        self.stateUpdateMatrix = np.array([ [1, 0, dt, 0], \
                                            [0, 1, 0, dt], \
                                            [0, 0, 1, 0], \
                                            [0, 0, 0, 1]])
        return self.stateUpdateMatrix


    def update(self, observation,  dt, observationCovariance=None ):
        if observationCovariance is not None:
            self.observationCovariance = observationCovariance
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

    def prediction(self, dt):
        '''
        prediction step
        Why split up? for multi object tracking, because sometimes
        we don't get an observation, we will not call the correction(), 
        but we still need to predict

        This step changes the state vector and the covariance matrix!

        so calling prediction() and correction() in a row 
        will give the same result as calling update()
        '''
        #Prediction step
        stateUpdateMatrix = self.getStateUpdateMatrix(dt)
        stateE = stateUpdateMatrix.dot(self.stateVector)
        stateCovarianceE = stateUpdateMatrix.dot(self.stateCovariance).dot(stateUpdateMatrix.T) + \
            self.stateTransitionCovariance

        self.stateVector = stateE
        self.stateCovariance = stateCovarianceE

        predictedObservation = self.observationMatrix.dot(self.stateVector)

        return self.stateVector, self.stateCovariance, predictedObservation

    def correction(self, observation, observationCovariance=None):
        '''
        correction step
        Why split up? for multi object tracking!
        '''
        if observationCovariance is not None:
            self.observationCovariance = observationCovariance
        #get back the estimation
        stateCovarianceE = self.stateCovariance
        stateE= self.stateVector
        #Generate Kalman Gain
        kalmanGain = stateCovarianceE.dot(self.observationMatrix.T).dot(np.linalg.inv(self.observationCovariance + \
            self.observationMatrix.dot(stateCovarianceE).dot(self.observationMatrix.T)))
        #Correct prediction
        self.stateVector = stateE + kalmanGain.dot(np.array(observation).T - self.observationMatrix.dot(stateE))
        self.stateCovariance = (np.eye(4) - kalmanGain.dot(self.observationMatrix)).dot(stateCovarianceE)
        return self.stateVector

    def getPrediction(self, dt):
        '''
        getPrediction step but don't change the state vector and the covariance matrix!
        '''
        #Prediction step
        stateUpdateMatrix = self.getStateUpdateMatrix(dt)
        stateE = stateUpdateMatrix.dot(self.stateVector)
        stateCovarianceE = stateUpdateMatrix.dot(self.stateCovariance).dot(stateUpdateMatrix.T) + \
            self.stateTransitionCovariance

        obsE = self.observationMatrix.dot(stateE)

        return stateE, stateCovarianceE, obsE

class ConstantVelocityConstantTurningRateFilter(BaseFilter):
    '''
    To C.K. :
    
    Fill all the questions first:

    The state space is defined as:
    [x, y, ?

    The observation space is defined as:
    [x, y]

    The state transition model is:
    x(k+1) = ?
    y(k+1) = ?
    twist(k+1) = twist(k) + turnRate(k) * dt
    turnRate(k+1) = turnRate(k) 
    It can only be represented as a bunch of equations.

    The observation model is:
    x(k) = x(k)
    y(k) = y(k)
    It is linear, that's why it can be represented as a matrix.
    '''
    def __init__(self, x=0, y=0, v=0, twist=0, turnRate=0, \
        stateNoise=0.5,observationNoise=10, id=None):
        #These are the state variables:
        self.stateVector = np.array([x, y, twist, v, turnRate]).T

    def stateTransition(self, stateVector, dt):
        '''
        stateTransition
        '''
        pass
        return stateVectorE

    def measurementFunction(self, stateVector):
        '''
        measurementFunction
        '''
        pass
        return predictedObservation

    def update(self, observation, dt):
        return super().update(observation, dt)
