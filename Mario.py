import torch
import torch.nn as nn
from torchvision import transforms as T 
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, [["right"], ["right", "A"]])

def Preprocess(observation):
     
     transform = T.Compose([
         T.ToPILImage(),
         T.Resize((84,84)),
         T.ToTensor()
     ])
     return transform(observation).unsqueeze(0)

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = FrameStack(env, 4)
obs = env.reset()
obs = np.array(obs)
processed_obs = Preprocess(obs)

class Deep_Q_net(nn.Module):
    def __init__(self):
        super.__init__()
        
        self.conv1 = nn.Conv2d(3,12,4,1),
        self.conv2 = nn.Conv2d(12,24,4,1),
        self.conv3 = nn.Conv2d(24,48,4,1),
        self.flatten = nn.Flatten(),
        self.fc1 = nn.Linear(48*210*160,64)
        self.fc2 = nn.Linear(64,64),
        self.fc3 = nn.linear(64,2)
    def forward(self,x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x