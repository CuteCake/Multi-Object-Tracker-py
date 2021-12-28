'''
An enviroment that generates a bunch of random points 
which can move around and appering and reappearing
in the window.

This enviroment can deal with non stable update frequency, look at self.clock.tick(60)

It also outputs the observation

It also outputs the visualization of the enviroment
The visualization is a pygame window
Red dots are the observation, white dots are the groud truth



Author: Zhihao
Dependency: pygame,  installation see:https://www.pygame.org/wiki/GettingStarted
'''
import random
import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

class Point:
    def __init__(self,posx,posy,twist,velocity,angular_velocity,pointID):
        self.posx = posx
        self.posy = posy
        self.twist = twist
        self.velocity = velocity
        self.pointID = pointID
        self.angular_velocity = angular_velocity


    def update(self,dt): #This is a constant velocity, constant turning rate model, this update is for ground truth generation
        self.posx += self.velocity*dt*math.cos(self.twist)
        self.posy += self.velocity*dt*math.sin(self.twist)
        self.angular_velocity += random.gauss(0,0.5)
        self.angular_velocity -= self.angular_velocity*0.1
        self.twist += self.angular_velocity*dt
        self.twist = self.twist%(2*math.pi)
        self.velocity += random.gauss(0,2)
        

    def getXY(self):
        return (self.posx,self.posy)
        
class PointsEnv:
    def __init__(self, width, height, numPoints, pointSize = 2, observation_noise = 5):
        self.width = width
        self.height = height
        self.numPoints = numPoints
        self.pointSize = pointSize
        self.points = []
        self.numPoints = numPoints
        self.pointSize = pointSize
        self.generatePoints()
        self.boxSizeX = width / 20
        self.boxSizeY = height / 20
        self.observation_noise = observation_noise
        self.dropOutProb = 0.2
        self.randomNoiseProb = 1 #Tthe probability of adding a random noise to the observation, It can be 
                                    # greater than 1, which adds more than 1 noise points
        self.clock = pygame.time.Clock()

    def generatePoint(self,id):
        point_a = Point(random.randrange(0,self.width), # initial x
            random.randrange(0,self.height),            # initial y
            random.uniform(0,2*math.pi),                # initial twist
            (random.uniform(0,10)+40),                   # initial velocity
            0 + random.gauss(0,1),                     # initial angular_velocity
            id)
        return point_a

    def generatePoints(self):
        for i in range(self.numPoints):
            self.points.append(self.generatePoint(i))

    def update(self): #Returns  observation
        dt = self.get_last_dt()
        for i in range(self.numPoints):
            self.points[i].update(dt)
        
        #check if the point is inside the frame
        for point in self.points:
            if point.posx < 0 or point.posx > self.width or point.posy < 0 or point.posy > self.height :
                pointID = point.pointID
                self.points[pointID] = self.generatePoint(pointID)
        
        self.clock.tick(60+random.randrange(-20,20)) #This limits The env to 60 frames per second by adding delay to the loop

        #return the observation as a 1D array
        observation = np.empty((self.numPoints,2))
        for i in range(len(self.points)):
            observation[i,0] = (self.points[i].posx + random.gauss(0,self.observation_noise))
            observation[i,1] = (self.points[i].posy + random.gauss(0,self.observation_noise))
        #delete an obervation randomly
        # if random.random() < 0.9:
        #     observation = np.delete(observation,random.randrange(0,len(observation)),axis=0)
        for _ in range(len(observation)):
            if random.random() < self.dropOutProb and len(observation) >0:
                observation = np.delete(observation,random.randrange(0,len(observation)),axis=0)

        # add random noise to the observation
        noise_volume = self.randomNoiseProb
        while(noise_volume > 0):
            if random.random() < noise_volume:
                noise = np.array([[random.randint(0,self.width),random.randint(0,self.height)]])
                observation = np.concatenate((observation,noise),axis=0)
                noise_volume -= 1
        return observation

    def get_last_dt(self):
        return self.clock.get_time()/1000.0

    def draw(self, screen):
        '''
        Draw the enviroment
        It draws the ground truth with a white dot
        It also draws a bounding box
        '''
        for point in self.points:
            pygame.draw.circle(screen, (255, 255, 255), point.getXY() , self.pointSize)
            #draw a box polygon around the point, rotate it by the point's twist
            #and draw it on the screen
            box = [(point.posx - self.boxSizeX, point.posy - self.boxSizeY),
                   (point.posx + self.boxSizeX, point.posy - self.boxSizeY),
                   (point.posx + self.boxSizeX, point.posy + self.boxSizeY),
                   (point.posx - self.boxSizeX, point.posy + self.boxSizeY)]
            box = np.array(box)

            box = box.astype(np.int32)
            #rotate the box using numpy, origin is the center of the box
            box_relative = np.dot(np.array([[math.cos(point.twist), -math.sin(point.twist)],
                                            [math.sin(point.twist), math.cos(point.twist)]]),
                                                            (box - box.mean(axis=0)).T )
            box = box_relative.T + box.mean(axis=0)
            pygame.draw.polygon(screen, (50, 200, 50), box, width=3)

    def draw_observed_points(self,screen,obsList):
        for i in range(len(obsList)):
            pygame.draw.circle(screen, (255, 10, 105), (int(obsList[i][0]),int(obsList[i][1])), self.pointSize)

    def draw_prediction(self,screen,prediction):
        # for i in range(len(prediction)):
        posx = int(prediction[0])
        posy = int(prediction[1])
        twist = prediction[2]

        pygame.draw.circle(screen, (10, 10, 255), (posx,posy), self.pointSize)

        box = [(posx - self.boxSizeX, posy - self.boxSizeY),
                (posx + self.boxSizeX, posy - self.boxSizeY),
                (posx + self.boxSizeX, posy + self.boxSizeY),
                (posx - self.boxSizeX, posy + self.boxSizeY)]
        box = np.array(box)

        box = box.astype(np.int32)
        #rotate the box using numpy, origin is the center of the box
        box_relative = np.dot(np.array([[math.cos(twist), -math.sin(twist)],
                                        [math.sin(twist), math.cos(twist)]]),
                                                        (box - box.mean(axis=0)).T )
        box = box_relative.T + box.mean(axis=0)
        pygame.draw.polygon(screen, (100, 100, 255), box, width=3)
        
if __name__ == "__main__": #This is for testing the enviroment
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    env = PointsEnv(640, 480, 10)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('time elapsed: ', env.get_time_elapsed())
                pygame.quit()
                quit()
        screen.fill((0, 0, 0))
        env.draw(screen)
        observation = env.update()
        env.draw_observed_points(screen, observation)
        # print(env.getObservation())
        pygame.display.update()