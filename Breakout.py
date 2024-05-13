from ale_py import ALEInterface
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

ale = ALEInterface()
from ale_py.roms import Breakout

ale.loadROM(Breakout)

import gymnasium as gym

env = gym.make('ALE/Breakout-v5',render_mode="human")
env.reset()
'''
for step in range(1000):
    env.render()
    random_action = env.action_space.sample()
    env.step(random_action)
env.close()
'''

WINDOW_LENGTH = 3
 

sequential_frame_buffer = []

temp_sequential_frames = deque(maxlen=WINDOW_LENGTH)

for i in range(10):
    if i == 1:
        action = 1
    else:
        action = 3
        
    
    observation, r, d, info, _ = env.step(action)  # Ignoring the extra value

    if len(temp_sequential_frames) == WINDOW_LENGTH:
        sequential_frame_buffer.append(list(temp_sequential_frames))
    temp_sequential_frames.append(observation)  

plt.imshow( sequential_frame_buffer[3][2]) 

