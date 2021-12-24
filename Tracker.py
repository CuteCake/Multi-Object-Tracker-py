#Test a multi-object tracker on the environment from environment.py
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

class trajectory:
    def __init__(self, x, y, theta,v,id):
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

    