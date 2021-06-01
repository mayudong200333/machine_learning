import numpy as np
import copy
from collections import deque
import gym
from gym import wrappers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

class PolicyNetwork(nn.Module):
    def __init__(self,num_state,num_action,hidden_size=16):
       super().__init__()
       self.fc1 = nn.Linear(num_state,hidden_size)
       self.fc2 = nn.Linear(hidden_size,hidden_size)
       self.fc3 = nn.Linear(hidden_size,num_action)

    def forward(self,x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        action_prob = F.softmax(self.fc3(h),dim=-1)
        return action_prob


class ReinforceAgent:
    def __init__(self,num_state,num_action,gamma=0.99,lr=0.001):
        self.num_state = num_state
        self.gamma = gamma
        self.pinet = PolicyNetwork(num_state,num_action).to(device)
        self.optimizer = optim.Adam(self.pinet.parameters(),lr=lr)
        self.memory = []

    def update_policy(self):
        R=0
        loss=0
        for r,prob in self.memory[::-1]:
            R = self.gamma*R + r
            loss -= torch.log(prob) * R
        loss /= len(self.memory)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_greedy_action(self,state):
        state_tensor = torch.tensor(state,dtype=torch.float).view(-1,self.num_state).to(device)
        action_prob = self.pinet(state_tensor.data).squeeze()
        action = torch.argmax(action_prob.data).item()
        return action

    def get_action(self,state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state).to(device)
        action_prob = self.pinet(state_tensor.data).squeeze()
        action = Categorical(action_prob).sample().item()
        return action,action_prob[action]

    def add_memory(self,r,prob):
        self.memory.append((r,prob))

    def reset_memory(self):
        self.memory = []


if __name__ == '__main__':
    num_episode = 600

    episode_rewards = []
    num_average_epidodes = 10

    env = gym.make('CartPole-v0')
    max_steps = env.spec.max_episode_steps

    agent = ReinforceAgent(env.observation_space.shape[0],env.action_space.n)

    for episode in range(num_episode):
        state = env.reset()
        episode_reward = 0
        for t in range(max_steps):
            action,prob = agent.get_action(state)
            next_state,reward,done,_ = env.step(action)
            episode_reward += reward
            agent.add_memory(reward,prob)
            state = next_state
            if done:
                agent.update_policy()
                agent.reset_memory()
                break
        episode_rewards.append(episode_reward)
        if episode % 20 == 0:
            print("Episode %d finished | Episode reward %f" % (episode, episode_reward))

        # 累積報酬の移動平均を表示
    moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes) / num_average_epidodes,
                                     mode='valid')
    plt.plot(np.arange(len(moving_average)), moving_average)
    plt.title('REINFORCE: average rewards in %d episodes' % num_average_epidodes)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.show()

    env.close()

    env = gym.make('CartPole-v0')
    frames = []
    for episode in range(5):
        state = env.reset()
        frames.append(env.render(mode='rgb_array'))
        done = False
        while not done:
            action = agent.get_greedy_action(state)
            state, reward, done, _ = env.step(action)
            frames.append(env.render(mode='rgb_array'))
    env.close()



