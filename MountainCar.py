import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym 
from collections import deque
import random

class QNet(nn.Module):
    def __init__(self,state_size,action_size):
        super(QNet,self).__init__()
        self.fc1 = nn.Linear(state_size,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,action_size)
    def forward(self,state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
        
deep_q_net = QNet(state_size,action_size)
env = gym.make('MountainCar-v0', render_mode="human")

action_size = env.action_space.n
state_size = env.observation_space.shape[0]
observation, info = env.reset()
deep_q_net = QNet(state_size,action_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(deep_q_net.parameters())

epochs = 20000
alpha = 0.8
gamma = 0.94
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001
replay_buffer_size = 10000
batch_size = 64

replay_buffer = deque(maxlen=replay_buffer_size)
 
for _ in range(epochs):
    observation,info = env.reset()
    state = torch.from_numpy(observation).float()
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample() 
        else:
            q_values = deep_q_net(state.unsqueeze(0))
            action = torch.argmax(q_values).item()

        next_observation,reward,terminated,truncated,info = env.step(action)
        next_state = torch.from_numpy(next_observation).float()
        done = terminated or truncated
        replay_buffer.append((state,action,reward,next_state,done))
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer,batch_size)
            states,actions,rewards,next_states,dones = zip(*batch)
            states = torch.stack(states)
            actions = torch.tensor(actions,dtype=torch.int64)
            rewards = torch.tensor(rewards,dtype=torch.float32)
            next_states = torch.stack(next_states)
            dones = torch.tensor(dones,dtype=torch.bool)
            q_values = deep_q_net(states).gather(1,actions.unsqueeze(1)).squeeze(1)
            next_q_values = deep_q_net(next_states).max(1)[0]
            next_q_values[dones] = 0.0
            target_q_values = rewards + gamma * next_q_values
            loss = criterion(q_values,target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        state = next_state
    epsilon = max(min_epsilon,epsilon-decay_rate)

env.close()
