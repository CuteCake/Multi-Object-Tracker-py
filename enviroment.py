# An enviroment that generates a bunch of random points 
# which can move around and appering and reappearing
# in the window.
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

    def update(self,dt):
        self.posx += self.velocity*dt*math.cos(self.twist)
        self.posy += self.velocity*dt*math.sin(self.twist)
        self.angular_velocity = random.gauss(0,0.1)
        self.twist += self.angular_velocity*dt
        self.twist = self.twist%(2*math.pi)
        

    def getXY(self):
        return (self.posx,self.posy)
        
class PointsEnv:
    def __init__(self, width, height, numPoints, pointSize = 2, observation_noise = 0.1):
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

    def generatePoints(self):
        for i in range(self.numPoints):
            self.points.append(self.generatePoint(i))

    def generatePoint(self,id):
        point_a = Point(random.randrange(0,self.width), # x
            random.randrange(0,self.height), # y
            random.uniform(0,2*math.pi), # twist
            (random.uniform(0,1)+1), # velocity
            0, # angular_velocity
            id)
        return point_a

    def draw(self, screen):
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
            box_relative = np.dot(box - box.mean(axis=0), np.array([[math.cos(point.twist), -math.sin(point.twist)],
                                                            [math.sin(point.twist), math.cos(point.twist)]]))
            box = box_relative + box.mean(axis=0)
            pygame.draw.polygon(screen, (100, 255, 100), box, width=3)



    def update(self,time):
        for i in range(self.numPoints):
            self.points[i].update(time)
        
        #check if the point is inside the frame
        for point in self.points:
            if point.posx < 0 or point.posx > self.width or point.posy < 0 or point.posy > self.height :
                pointID = point.pointID
                self.points[pointID] = self.generatePoint(pointID)

    def getObservation(self):
        #return the observation as a 1D array
        observation = np.empty((self.numPoints,2))
        for i in range(len(self.points)):
            observation[i,0] = (self.points[i].posx + random.gauss(0,self.observation_noise))
            observation[i,1] = (self.points[i].posy + random.gauss(0,self.observation_noise))
        return observation

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    env = PointsEnv(640, 480, 10)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        screen.fill((0, 0, 0))
        env.draw(screen)
        env.update(0.03)
        print(env.getObservation())
        pygame.display.update()