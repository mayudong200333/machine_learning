import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

# Gymは、Open AIのRL用ツールキットです
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# OpenAI Gym用に使うNES エミュレーター
from nes_py.wrappers import JoypadSpace

#OpenAI Gymのスーパー・マリオ・ブラザーズの環境
import gym_super_mario_bros



device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

import matplotlib.pyplot as plt


class SkipFrame(gym.Wrapper):
    def __init__(self,env,skip):
        super().__init__(env)
        self._skip = skip

    def step(self,action):
        total_reward=0.0
        done = False
        for i in range(self._skip):
            obs,reward,done,info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs,total_reward,done,info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self,env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0,high=255,shape=obs_shape,dtype=np.uint8)

    def permute_orientation(self,observation):
        observation = np.transpose(observation,(2,0,1))
        observation = torch.tensor(observation.copy(),dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self,env,shape):
        super().__init__(env)
        if isinstance(shape,int):
            self.shape = (shape,shape)
        else:
            self.shape = tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0,high=255,shape=obs_shape,dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


class ReplayBuffer:
    def __init__(self,capacity):
        self.memory = deque(maxlen=capacity)
        self.use_cuda = torch.cuda.is_available()

    def cache(self,state,next_state,action,reward,done):
        state = state.__array__()
        next_state = next_state.__array__()
        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor([action], dtype=torch.float)
            reward = torch.tensor([reward])
            done = torch.tensor([done])
        self.memory.append((state,next_state,action,reward,done))

    def sample(self,batch_size):
        batch = random.sample(self.memory,batch_size)
        state,next_state,action,reward,done = map(torch.stack,zip(*batch))
        return state,next_state,action.squeeze(),reward.squeeze(),done.squeeze()


class MarioNet(nn.Module):
    """
    単純なCNN構造とし、以下の通りです
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target のパラメータは固定されます
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

class MarioAgent:
    def __init__(self,state_dim,action_dim,gamma=0.99,lr=1e-3,batch_size=32,memory_size=50000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = MarioNet(self.state_dim,self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(memory_size)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update(self):
        state,next_state,action,reward,done = self.replay_buffer.sample(self.batch_size)
        q_e = self.td_estimate(state,action)
        q_t = self.td_target(reward,next_state,done)
        loss = self.loss_fn(q_e,q_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.net.target.load_state_dict(self.net.online.state_dict())

    def act(self,state,episode):
        epsilon = 0.7* (1/(episode+1))
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state.__array__()
            state = torch.tensor(state).cuda()
            state = state.unsqueeze(0)
            action_values = self.net(state,model="online")
            action_idx = torch.argmax(action_values,axis=1).item()
        return action_idx


if __name__ == '__main__':
    # 各種設定
    num_episode = 50  # 学習エピソード数（学習に時間がかかるので短めにしています）
    memory_size = 10000  # replay bufferの大きさ
    initial_memory_size = 100  # 最初に貯めるランダムな遷移の数
    # ログ用の設定
    episode_rewards = []
    num_average_epidodes = 5

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env,skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env,shape=84)
    env = FrameStack(env,num_stack=4)

    max_steps = env.spec.max_episode_steps
    agent = MarioAgent(state_dim=(4,84,84),action_dim=env.action_space.n,memory_size=memory_size)
    state = env.reset()
    for step in range(initial_memory_size):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.cache(state,next_state,action,reward,done)
        state = env.reset() if done else next_state

    for episode in range(num_episode):
        state = env.reset()
        episode_reward = 0
        for t in range(max_steps):
            action = agent.act(state, episode)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            agent.replay_buffer.cache(state,next_state,action,reward,done)
            agent.update()
            state = next_state
            if done:
                break
        episode_rewards.append(episode_reward)
        if episode % 2 == 0:
            print("Episode %d finished | Episode reward %f" % (episode, episode_reward))

    moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes) / num_average_epidodes, mode='valid')
    plt.plot(np.arange(len(moving_average)), moving_average)
    plt.title('DQN: average rewards in %d episodes' % num_average_epidodes)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.show()

    env.close()






