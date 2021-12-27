# Multi object tracking test platform
### An Multi object tracking agrithm in python
The kalman filter is in motionModels.py.
The tracker is in tracker.py, 
### An enviroment that generates a bunch of random points 
which can move around and appering and reappearing
in the window.

This enviroment can deal with non stable update frequency, look at self.clock.tick()
This enviroment also deals with dropping observation and adding random noise to the observation.
This enviroment also deals with adding random false positive observation.

It also outputs the visualization of the enviroment using a pygame window

Red dots are the observation, white dots are the groud truth, other dots are the prediction

### Dependency: 
- scipy
- pygame,  installation see:https://www.pygame.org/wiki/GettingStarted

### How to use:
open tracker.py, click run

Author: Zhihao