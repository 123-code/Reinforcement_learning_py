import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import math 
import random 
import matplotlib
import matplotlib.pyplot as plt
from collections import deque,namedtuple
from itertools import count

device = "cuda"
print(device)


env = gym.make('ALE/Breakout-v5')

'''
cnns que toman la diferencia entre screens actuales y anteriores 
'''

class DeepQNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DeepQNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 210 * 160, 64) 
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 210, 160) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# numero de parametros que pasan por la red antes de cambiar los pesos
BATCH_SIZE = 64
EPOCHS = 100
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 10000 
EPSILON = 1.0
MAX_EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY_RATE = 0.001
TAU = 0.005 # update rate de la red target
LR = 1e-4 # tasa de aprendizaje

replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n 
deep_q_net = DeepQNet(state_size,action_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(deep_q_net.parameters(),lr=LR)

for x in range(EPOCHS):
    observation,info = env.reset()
    state = torch.from_numpy(observation).float().to(device) 
    done = False
    while not done:
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            q_values = deep_q_net(state.unsqueeze(0))
            action = torch.argmax(q_values).item()
        next_observation,reward,terminated,truncated,info = env.step(action)
        next_state = torch.from_numpy(next_observation).float().to(device)
        done = terminated or truncated
        replay_buffer.append((state,action,reward,next_state,done))
        if len(replay_buffer)>= BATCH_SIZE:
            
            batch = random.sample(replay_buffer,BATCH_SIZE)
            states,actions,rewards,next_states,dones = zip(*batch)
            states = torch.stack(states).to(device)
            actions = torch.tensor(actions,dtype=torch.int64).to(device)
            rewards = torch.tensor(rewards,dtype=torch.float32).to(device)
            next_states = torch.stack(next_states).to(device)
            dones = torch.tensor(dones,dtype=torch.bool).to(device)
            q_values = deep_q_net(states).gather(1,actions.unsqueeze(1)).squeeze(1)
            next_q_values = deep_q_net(next_states).max(1)[0]
            next_q_values[dones] = 0.0
            target_q_values = rewards + GAMMA* next_q_values
            loss = criterion(q_values,target_q_values)
            print(f"loss:{loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        step = next_state
    epsilon = max(MIN_EPSILON,EPSILON-DECAY_RATE)    
env.close() 