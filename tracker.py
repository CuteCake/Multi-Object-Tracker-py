#Test a multi-object tracker on the environment from environment.py
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import pygame
from enviroment import Point, PointsEnv
from motionModel import ConstantVelocityFilter, ConstantVelocityConstantTurningRateFilter

from itertools import permutations


class Track: #This is a class for a track, which is tracking a single object using some filter
    '''
    This is a Track class that can automatically do management itself

    It have a timer of how long it has not recived observations, and how long it has not recived observations
    Use a upper level class to get the .isDead and .isConfirmedTrack to update the list of tracks

    To delete a track, delete the reference of the track from the upper level class, 
    hope the garbage collector will do the job
    '''
    def __init__(self,observation = None, motion_model = 'constant_velocity', track_id=None,
     time_to_confirm = 0.3, #time to confirm a track
     time_to_kill = 0.3):  #time to kill a track if not recived observations
        assert observation is not None

        if motion_model == 'constant_velocity': #set initial state from observation
            self.filter = ConstantVelocityFilter(
                x = observation[0],
                y = observation[1],
                vx = 0,
                vy = 0)
        elif motion_model == 'constant_velocity_constant_turning_rate':
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

        #The following part is for track maintaince
        self.track_id = track_id
        self.time_recived_observations = 0
        self.time_not_recived_observations = 0
        self.time_to_confirm = time_to_confirm
        self.isConfirmedTrack = False
        self.time_to_kill = time_to_kill
        self.isDead = False
        self.closest_ob = None

        print('Track created with id:', self.track_id)
    
    def __del__(self):
        print('Track', self.track_id, 'deleted')

    def doGating(self, obs, dt, obsCov=None) -> None:
        stateE, stateCovarianceE, ob_E = self.filter.getPrediction(dt)
        self.gated_obs = []
        best_dist = float('inf')
        
        for ob in obs:
            ob = ob
            distance = self._euclidean_distance(ob, ob_E)
            # distance = self._mahalanobis_distance(ob, ob_E, obsCov)
            if distance < 20:
                self.gated_obs.append(ob)
                if distance < best_dist:
                    best_dist = distance
                    self.closest_ob = ob
        if len(self.gated_obs) == 0:
            self.closest_ob = None
            self.gated_obs = None
            # print('No valid obs in id: '+ str(self.track_id))

        return self.gated_obs


    def doPredictionStep(self, dt) -> np.array:
        return self.filter.prediction(dt) #return the state vector and covariance

    def doCorrectionStep(self, observation, obsCov=None) -> None:
        stateVector = self.filter.correction(observation, observationCovariance=obsCov)
        return stateVector

    def doMaintenance(self, dt = None, observation = None) -> None:
        assert dt is not None

        if observation is None:
            self.time_not_recived_observations += dt
            self.time_recived_observations = 0
        else:
            self.time_recived_observations += dt
            self.time_not_recived_observations = 0

        #Update the status of the track
        if self.time_recived_observations > self.time_to_confirm:
            self.isConfirmedTrack = True
        if self.time_not_recived_observations > self.time_to_kill:
            self.isDead = True

    def getState(self) -> np.array:
        return self.filter.stateVector

    def getStateCovariance(self) -> np.array:
        return self.filter.stateCovariance

    def _euclidean_distance(self, obs, obs_predicted):
        '''
        Calculate the euclidean distance between observation and predicted observation
        Lets assume the observation is a vector, and the predicted observation is a vector
        '''
        D = np.linalg.norm(obs-obs_predicted)

        return D

    def _mahalanobis_distance(self, obs, obs_predicted, obsPCov):
        '''
        Calculate the Mahalanobis distance between observation and predicted observation
        Lets assume the observation is a vector, and the predicted observation is a vector
        '''
        D = math.sqrt((obs-obs_predicted).T.dot(np.linalg.inv(obsPCov)).dot(obs-obs_predicted))

        return D

    def update(self, observation, dt, obsCov=None, doDeadReckoning=False) -> None:#This is abandonded
        
        #Do the maintenance of the track
        #Update the timers of the track
        if observation is None:
            doDeadReckoning = True
            self.time_not_recived_observations += dt
            self.time_recived_observations = 0
        else:
            self.time_recived_observations += dt
            self.time_not_recived_observations = 0

        #Update the status of the track
        if self.time_recived_observations > self.time_to_confirm:
            self.isConfirmedTrack = True
        if self.time_not_recived_observations > self.time_to_kill:
            self.isDead = True

        #Do the update
        stateVector = self.filter.update(observation, dt, observationCovariance=obsCov, doDeadReckoning=doDeadReckoning)
        return stateVector

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
        state_noise = 5, motion_model = 'constant_velocity'):
        self.sensor_noise = sensor_noise
        self.measurement_noise = measurement_noise
        self.state_noise = state_noise

        #The following are for logging purposes
        self.state_estimate = []
        self.state_estimate_history = []

        self.measurement_history = []
        self.prediction_history = []


        if motion_model == 'constant_velocity':
            self.motionFilter = ConstantVelocityFilter()
        elif motion_model == 'constant_turning_rate':
            self.motionFilter = ConstantVelocityConstantTurningRateFilter()
        else:
            raise Exception('Measurement model not supported')

    def updateTracker(self, observation, dt):
        print('dt: ', dt)
        self.state_estimate = self.motionFilter.update(observation, dt)

        return self.state_estimate

class MultiTracker(BaseTracker):
    '''
    A multi-object Kalman Filter tracker

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

        Delete: When a track is not detected for a continuous of some time
                , delete the track
                When a track is too simillar to another track, delete the track
        
    3. update the Kalman Filter for each object
        Method 1: Use a single filter (KF,EKF,UKF, on some motion model)
        Method 2: Use a IMM Interactive Multi-Object Kalman Filter
              It updates one object using several models then outputs a weighted sum

    4. do gating (measurement validation)
        This is mainly for JPDA
        if the measurement is not in the gate range, don't waste the computation

    5. retun the state_estimate
    '''
    def __init__(self, obs = None, sensor_noise = 5, measurement_noise = 5, 
        state_noise = 5, motion_model = 'constant_velocity',
        association_method = 'GNN'):
        self.sensor_noise = sensor_noise
        self.measurement_noise = measurement_noise
        self.state_noise = state_noise

        #The following are for logging purposes
        self.state_estimate = []
        self.state_estimate_history = []

        self.measurement_history = []
        self.prediction_history = []

        self.motion_model = motion_model
        self.association_method = association_method
        # self.tracked_objects = [] #the list for tracks, including confirmed and tentative tracks
        self.tracked_objects_dict = {} #the dict for tracks, including confirmed and tentative tracks
        self.next_track_id = 0

        #Initialize trackers
        assert obs is not None
        assert len(obs) > 1
        for i in range(len(obs)):
            self._createTrack(obs[i])

        
    def updateTracker(self, observations, dt, obsCov=None):
        
        '''
        observations -> [[x,y],[x,y],...] is a 2D array
        state_estimate -> [[x,y,vx,vy],[x,y,vx,vy],...]
        trackedObjects -> [TrackedObject,TrackedObject,...]
        '''

        #Do gating before association
        #Call prediction step for every track's filter, get the new state_estimate for association
        #1. data association 

        for track in self.tracked_objects_dict.values():
            track.doGating(observations, dt)
        '''
        if self.association_method == 'GNN':
            self._GNN_data_association(observation, dt, obsCov=obsCov)
        else:
            raise Exception('Association method not supported')
        
        #2. track management
        self._deleteDeadTracks()

        #3. track update
        for track in self.tracked_objects:
            track.doCorrectionStep(observation, dt, obsCov) #TODO
        '''

        #Test:
        for track in self.tracked_objects_dict.values():
            track.doPredictionStep(dt)
            if track.closest_ob is not None:
                track.doCorrectionStep(track.closest_ob)
            track.doMaintenance(dt=dt, observation=track.closest_ob)

        for ob in observations:
            notAssociated = True
            for track in self.tracked_objects_dict.values():
                if np.array_equal(ob,  track.closest_ob):
                    notAssociated = False
                    break
            if notAssociated:
                self._createTrack(ob)


        self._deleteDeadTracks()
        return self.tracked_objects_dict

    def _createTrack(self, observation):
        '''
        Create a new track for the observation
        '''
        self.next_track_id += 1
        # self.tracked_objects.append(Track(observation = observation, \
        #     motion_model=self.motion_model, track_id=self.next_track_id))
        self.tracked_objects_dict[self.next_track_id] = Track(observation = observation, \
            motion_model=self.motion_model, track_id=self.next_track_id)
        return self.next_track_id

    def _deleteDeadTracks(self):
        '''
        Delete a track
        '''
        # for index, track in enumerate(self.tracked_objects):
        #     if track.isDead:
        #         self.tracked_objects.pop(index) #delete the track from the list, 
        #         #and the track will be deleted automatically
        #         #potential bug: the index of the list will be changed??

         #Method 2: using the dict:
        id_to_delete = []
        for id,track in self.tracked_objects_dict.items():
            if track.isDead:
                id_to_delete.append(id)
        for id in id_to_delete:
            self.tracked_objects_dict.pop(id)

        return

    def _GNN_data_association(self, obs, obs_predicted, track_ids, obsCov=None):
        '''
        Data association using GNN

        we get every observation a track, 
        if an existing track is not found,
        we create a new track for it

        observation -> [[x,y],[x,y],...]

        return:
            association list -> [track_id, track_id,...]
             the length of the list is the number of observations

        '''
        #This is a brute force method: going through every possible association,
        #Check is the combination is valid, if yes, 
        #calculate euclidean distance, TODO use Mahalanobis distance
        #And then add to the list
        min_sum_dist = np.inf
        permut = permutations(list(range(len(obs))),list(range(len(obs_predicted))))
        for combination in permut:
            sum_dist = 0
            #calculate the sum of euclidean distance
            gate_checking = False
            for i in range(len(combination)):
                pass
            #check if the sum of distance is better
            if sum_dist < min_sum_dist:
                min_sum_dist = sum_dist
                best_comnination = combination

        #Calculate the num_tracks by num_observations cost matrix
        # num_tracks = len(obs)
        # num_observations = len(obs_predicted)
        # cost_matrix = np.zeros((num_tracks, num_observations))
        # for i in range(num_tracks):
        #     for j in range(num_observations):
        #         cost_matrix[i,j] = np.linalg.norm(obs[i]-obs_predicted[j])
        # use kuhn munkres algorithm to find the best association


        return best_comnination[track_ids]

    def _kuhn_munkres(self, cost_matrix):
        pass #TODO

    def _JPDA_data_association(self, observation, dt, obsCov=None):
        raise NotImplementedError

    def _mahalanobis_distance(self, obs, obs_predicted, obsPCov):
        '''
        Calculate the Mahalanobis distance between observation and predicted observation

        Lets assume the observation is a vector, and the predicted observation is a vector
        '''
        D = math.sqrt((obs-obs_predicted).T.dot(np.linalg.inv(obsPCov)).dot(obs-obs_predicted))

        return D
    
    def _2D_mahalannobis_threshold_from_probability(self, prob, Dim):
        '''
        Calculate the threshold for Mahalanobis distance from probability 
        that observation is valid
        '''

        T = math.sqrt(-2*math.log(1-prob)) #only valid for Dim = 2

        return T



    # def dataAssociation(self, observations, TrackedObjects):
    #     '''
    #     observations -> [[x,y],[x,y],...]
    #     state_estimate -> [[x,y,vx,vy],[x,y,vx,vy],...]
    #     trackedObjects -> [TrackedObject,TrackedObject,...]
    #     '''
    #     if self.association_method == 'GNN':
    #         #TODO
    #         return self.GNN(observations, state_estimate)
    #     else:
    #         raise Exception('Association method not supported')

    # def trackManagement(self, observations, TrackedObjects):
    #     pass

    # def updateTracker(self, observation, dt,):

    #     return self.state_estimate

    # def gating(self, observations, state_estimate):
    #     pass
    


if __name__ == "__main__":
    # pygame.init()
    # screen = pygame.display.set_mode((640, 480))
    # env = PointsEnv(640, 480, 10)
    # tracker = SingleTracker()
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             print(env.get_time_elapsed())
    #             pygame.quit()
    #             quit()
    #     screen.fill((0, 0, 0))
    #     env.draw(screen)
    #     observation = env.update()
    #     env.draw_observed_points(screen, observation)
        
    #     obs = np.array(observation[0])
    #     stateVec = tracker.updateTracker(obs, env.get_last_dt())
    #     # env.draw_prediction(screen, stateVec)
    #     pygame.draw.circle(screen, (10,10, 255), (int(stateVec[0]),int(stateVec[1])), 5)
    #     pygame.display.update()
    
    import time

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    env = PointsEnv(640, 480, 10)
    observation = env.update()
    tracker = MultiTracker(obs=observation)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        screen.fill((0, 0, 0))
        env.draw(screen)
        observation = env.update()
        env.draw_observed_points(screen, observation)
        obs = np.array(observation)
        # start_time = time.time()

        # The call to update tracker, input is a 2D array of observation, 
        # and the output is a array of Track objects
        objects_dict = tracker.updateTracker(obs, env.get_last_dt())
        # print('fps:', 1/(time.time()-start_time))
        # env.draw_prediction(screen, stateVec)
        for object in objects_dict.values():
            pygame.draw.circle(screen, object.color, (int(object.getState()[0]),int(object.getState()[1])), 4)
        pygame.display.update()
