import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

learning_rate = 0.01
gamma = 0.99


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)
        self.gamma = gamma

        self.policy_history = torch.Tensor()
        self.reward_episode = []
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)


policy = Policy()
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

# select action to perform, run the policy network on the state, then get a probability distribution over the actions. finally sample from there
def select_action(state):
    state_array = state[0]
    state = torch.tensor(state_array).float()  # Convert state to a PyTorch tensor
    state = state.unsqueeze(0) 
    action_probs = policy(state)  # Pass the state through the policy network
    c = torch.distributions.Categorical(probs=action_probs)  
    action = c.sample()

    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action).unsqueeze(0)])

    else:
        policy.policy_history = c.log_prob(action).unsqueeze(0)

    return action


def update_policy():
    R = 0
    rewards = []
    # reward discounting
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    loss = -torch.sum(torch.mul(policy.policy_history, rewards)) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = torch.Tensor()
    policy.reward_episode = []


def main(episodes):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset()
        done = False

        for time in range(1000):
            action = select_action(state)
            state, reward, terminated, truncated, info = env.step(action.item()) 
            state = torch.tensor(state).float().unsqueeze(0)
         
            done = terminated or truncated
            policy.reward_episode.append(reward)

            if done:
                break
        running_reward = (running_reward * 0.99) + (time * 0.01)
        update_policy()

        if episode % 50 == 0:
            print("Episode {}\tLast length: {:5d}\tAverage length: {:.2f}".format(episode, time, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward,
                                                                                                          time))
            break


episodes = 1000
main(episodes)
