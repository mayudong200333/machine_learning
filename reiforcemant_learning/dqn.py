import gym
import math
import random
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# matplotlibの設定
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
  from IPython import display

plt.ion()

# gpuが使用される場合の設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

#Replay Memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','done'))

class ReplayMemory:
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self,*args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1) % self.capacity

    def sample(self,batch_size):
        batches = random.sample(self.memory,batch_size)
        states = np.array([batches[i].state.tolist() for i in range(batch_size)])
        actions = np.array([batches[i].action for i in range(batch_size)])
        next_states = np.array([batches[i].next_state.tolist() for i in range(batch_size)])
        rewards = np.array([batches[i].reward for i in range(batch_size)])
        dones = np.array([batches[i].done for i in range(batch_size)])
        return {'states': states, 'next_states': next_states, 'rewards': rewards, 'actions': actions,'dones':dones}

    def __len__(self):
        return len(self.memory)

#dqn algorithm
class QNetwork(nn.Module):
    def __init__(self,num_state,num_action,hidden_size=16):
        super().__init__()
        self.fc1 = nn.Linear(num_state,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.fc4 = nn.Linear(hidden_size,num_action)

    def forward(self,x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        h = F.elu(self.fc3(h))
        y = F.elu(self.fc4(h))
        return y

class DqnAgent:
    def __init__(self,num_state,num_action,gamma=0.99,lr=0.001,batch_size=32,memory_size=50000):
        self.num_state = num_state
        self.num_action = num_action
        self.gamma = gamma
        self.batch_size = batch_size
        self.qnet = QNetwork(num_state,num_action)
        self.target_qnet = copy.deepcopy(self.qnet)
        self.optimizer = optim.Adam(self.qnet.parameters())
        self.replay_buffer = ReplayMemory(memory_size)

    def update_q(self):
        batch = self.replay_buffer.sample(self.batch_size)
        q = self.qnet(torch.tensor(batch['states'],dtype=torch.float))
        targetq = copy.deepcopy(q.data.numpy())
        maxq = torch.max(self.target_qnet(torch.tensor(batch['next_states'],dtype=torch.float)),dim=1).values
        for i in range(self.batch_size):
            targetq[i,batch['actions'][i]] = batch['rewards'][i] + self.gamma * maxq[i] * (not batch['dones'][i])
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(q,torch.tensor(targetq))
        loss.backward()
        self.optimizer.step()
        self.target_qnet = copy.deepcopy(self.qnet)

    def get_greedy_action(self,state):
        state_tensor = torch.tensor(state,dtype=torch.float).view(-1,self.num_state)
        action = torch.argmax(self.qnet(state_tensor).data).item()
        return action

    def get_action(self,state,episode):
        epsilon = 0.7* (1/(episode+1))
        if epsilon <= np.random.uniform(0.1):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.num_action)
        return action

#学習
num_episode = 300
memory_size = 10000
initial_memory_size = 500

episode_rewards = []
num_average_episodes = 10

env = gym.make('CartPole-v0')
max_steps = env.spec.max_episode_steps

agent = DqnAgent(env.observation_space.shape[0],env.action_space.n,memory_size=50000)

state = env.reset()
for step in range(initial_memory_size):
    action = env.action_space.sample()
    next_state,reward,done,_ = env.step(action)
    agent.replay_buffer.push(state,action,next_state,action,done)
    state = env.reset() if done else next_state

for episode in range(num_episode):
    state = env.reset()
    episode_reward = 0
    for t in range(max_steps):
        action = agent.get_action(state,episode)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.replay_buffer.push(state,action,next_state,action,done)
        agent.update_q()
        state = next_state
        if done:
            break
    episode_rewards.append(episode_reward)
    if episode % 20 == 0:
        print("Episode %d finished | Episode reward %f" % (episode, episode_reward))

moving_average = np.convolve(episode_rewards, np.ones(num_average_episodes)/num_average_episodes, mode='valid')
plt.plot(np.arange(len(moving_average)),moving_average)
plt.title('DQN: average rewards in %d episodes' % num_average_episodes)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.show()

env.close()







