#Test a multi-object tracker on the environment from environment.py
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import pygame
from pygame.math import disable_swizzling
from enviroment import Point, PointsEnv
from motionModel import ConstantVelocityFilter, ConstantVelocityConstantTurningRateFilter

from itertools import permutations, product


class Track: #This is a class for a track, which is tracking a single object using some filter
    '''
    This is a Track class that can automatically do management itself

    It have a timer of how long it has not recived observations, and how long it has not recived observations
    Use a upper level class to get the .isDead and .isConfirmedTrack to update the list of tracks

    To delete a track, delete the reference of the track from the upper level class, 
    hope the garbage collector will do the job
    '''
    def __init__(self,observation = None, motion_model = 'constant_velocity', track_id=None,
     time_to_confirm = 0.2, #time to confirm a track
     time_to_kill = 0.2):  #time to kill a track if not recived observations
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
        self.ob_E = ob_E
        self.gated_obs = []
        # best_dist = float('inf')
        # for ob in obs:
        #     ob = ob
        #     distance = self._euclidean_distance(ob, ob_E)
        #     # distance = self._mahalanobis_distance(ob, ob_E, obsCov)
        #     if distance < 20:
        #         self.gated_obs.append(ob)
        #         if distance < best_dist:
        #             best_dist = distance
        #             self.closest_ob = ob
        # if len(self.gated_obs) == 0:
        #     self.closest_ob = None
        #     self.gated_obs = None
        #     # print('No valid obs in id: '+ str(self.track_id))

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

    # def isInGate(self, obs, dt, obsCov=None) -> bool: #Not used
    #     '''
    #     Check if the observation is in the gate
    #     '''
        
    #     distance = self._euclidean_distance(obs, self.ob_E)
    #     if distance < 20:
    #         valid = True
    #     else:
    #         valid = False
    #     print('Distance: ', distance)
    #     return valid, distance

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

        self.max_track_num = 1000

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
        track_ids_assod, obs_assod, obs_not_assod = self._GNN_data_association(observations, self.tracked_objects_dict, dt)

        for track in self.tracked_objects_dict.values():
            track.doPredictionStep(dt)

        for track_id, ob in zip(track_ids_assod, obs_assod):
            self.tracked_objects_dict[track_id].doCorrectionStep(ob)
            self.tracked_objects_dict[track_id].doMaintenance(dt=dt, observation=ob)

        for id in self.tracked_objects_dict.keys():
            if id not in track_ids_assod:
                self.tracked_objects_dict[id].doMaintenance(dt=dt, observation=None)

        for ob in obs_not_assod:
            self._createTrack(ob)

        self._deleteDeadTracks()

        self.validTracks = {}
        for track in self.tracked_objects_dict.values():
            if track.isConfirmedTrack:
                self.validTracks[track.track_id] = track
        return self.validTracks

    def _createTrack(self, observation):
        '''
        Create a new track for the observation
        1, Check if max_track_num is reached
        2. Avoid track id number overflow
        3. Avoid duplicate track id number
        4. Create a new track
        '''
        if len(self.tracked_objects_dict) > self.max_track_num:
            return None
            #raise Exception('Maximum number of tracks reached')

        self.next_track_id += 1
        if self.next_track_id > self.max_track_num*10: #This is to avoid overflow
            self.next_track_id = 0
        while self.next_track_id in self.tracked_objects_dict.keys(): #Track ID already exists
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

    def _GNN_data_association(self, obs, track_dict, dt, obsCov=None):
        '''
        Data association using Global Nearest Neighbor

        input:
            track_dict: {track_id: Track}
            observation -> [[x,y],[x,y],...]

        return:
            associated_tracks -> [track_id, track_id,...]
            associated_observations -> [observation, observation,...] , The same order as associated_tracks

            not_associated_observations -> [observation, observation,...]

        '''
        #1. Calculate the num_tracks by num_observations cost matrix
        track_id_list = list(track_dict.keys()) #track_id_list = [1,9,3,...]
        track_id_list.sort() #sort track id to reduce the chance of duplicate tracks for the same observation
        cost_matrix = np.zeros((len(track_id_list), len(obs))) #(num_tracks, num_observations)
        for i in range(len(track_id_list)):
            for j in range(len(obs)):
                cost_matrix[i,j] = np.linalg.norm(obs[j]-track_dict[track_id_list[i]].ob_E)

        #2. Find the best association
        row_ind, col_ind = self._kuhn_munkres(cost_matrix)
        track_ind = list(row_ind) #the index in the track_id_list
        obs_ind = list(col_ind)   #the index in the obs

        # Transform the indices back into a list of track_id and a list of obs ids
        # Also, only add if it passes the gating check
        associated_track_ids = []
        associated_obs_ids = []

        for track_i, obs_i in zip(track_ind, obs_ind):
            cost = cost_matrix[track_i, obs_i]

            if cost < 50:
                associated_track_ids.append(track_id_list[track_i])
                associated_obs_ids.append(obs_i)
                # assert np.linalg.norm(obs[obs_i]-track_dict[track_id_list[track_i]].ob_E) < 20
                # print('track_id: ', track_i, 'dist: ', np.linalg.norm(obs[obs_i]-track_dict[track_id_list[track_i]].ob_E))
        
        # Turn obs_ind into a list of acturall observations
        associated_obs = []
        for obs_id in associated_obs_ids:
            associated_obs.append(obs[obs_id])

        #Last, lets get the obs which are not associated, so we can create new tracks for them
        not_associated_obs = []
        for i, ob in enumerate(obs):
            if i not in associated_obs_ids:
                not_associated_obs.append(ob)
                
        # print('not_associated_obs: ', len(not_associated_obs))
        assert len(associated_track_ids) == len(associated_obs)
        assert len(not_associated_obs) == len(obs) - len(associated_obs)
        # print('finish data association')
        # for id ,ob in zip(associated_track_ids, associated_obs):
        #     print('track_id: ', id, 'obs: ', ob, 'dist: ', np.linalg.norm(ob-track_dict[id].ob_E))

        return associated_track_ids, associated_obs, not_associated_obs


        # print('row_ind: ', row_ind)
        # print('col_ind: ', col_ind)
        # min_sum_dist = np.inf
        
        # permut = permutations(list(range(len(obs))),len(track_id_list))
        # best_comnination = None
        # for combination in permut:
        #     zipped = zip(combination, track_id_list)
        #     sum_dist = 0
        #     #calculate the sum of euclidean distance
        #     valid = True
        #     for pair in zipped:
        #         # print('pair: ', pair)
        #         valid, dist = track_dict[pair[1]].isInGate(obs[pair[0]],dt)
        #         if not valid:
        #             valid = False
        #         sum_dist += dist
        #     #check if the sum of distance is better
        #     if sum_dist < min_sum_dist and valid:
        #         min_sum_dist = sum_dist
        #         best_comnination = combination



    def _kuhn_munkres(self, cost_matrix):
        import scipy.optimize as op
        row_ind, col_ind = op.linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

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
        # env.draw(screen)
        observation = env.update()
        env.draw_observed_points(screen, observation)
        obs = np.array(observation)
        start_time = time.time()

        # The call to update tracker, input is a 2D array of observation, 
        # and the output is a array of Track objects
        objects_dict = tracker.updateTracker(obs, env.get_last_dt())
        # print('fps:', 1/(time.time()-start_time))
        # env.draw_prediction(screen, stateVec)
        # print('sum of track id: ', len(objects_dict.keys()))
        for object in objects_dict.values():
            pygame.draw.circle(screen, object.color, (int(object.getState()[0]),int(object.getState()[1])), 4)
        pygame.display.update()
