# Multi object tracking

Demo video: https://www.youtube.com/watch?v=-E1bOhiTmi0

### A multi object tracking agrithm in python
The kalman filter is in motionModels.py.
The tracker is in tracker.py, 
### An enviroment that generates a bunch of points 
Points are generated using constant velocity constant turning model.

This enviroment can generate non-stable update frequency, look at self.clock.tick()
This enviroment can generate dropping observation and adding random noise to the observation.
This enviroment can generate random false positive observation.

It also outputs the visualization of the enviroment using a pygame window

Red dots are the observation, white dots are the groud truth, other dots are the prediction

### Dependency: 
- scipy
- pygame,  installation see:https://www.pygame.org/wiki/GettingStarted

### How to use:
open tracker.py, click run

Author: Zhihao