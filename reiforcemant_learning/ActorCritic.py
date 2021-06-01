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

class ActorCriticNetwork(nn.Module):
    def __init__(self,num_state,num_action,hidden_size=16):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state,hidden_size)
        self.fc2a = nn.Linear(hidden_size,num_action)
        self.fc2c = nn.Linear(hidden_size,1)

    def forward(self,x):
        h = F.elu(self.fc1(x))
        action_prob = F.softmax(self.fc2a(h))
        state_value = self.fc2c(h)
        return action_prob,state_value


class ActorCriticAgent:
    def __init__(self,num_state,num_action,gamma=0.9,lr=0.001):
        self.num_state = num_state
        self.gamma = gamma
        self.acnet = ActorCriticNetwork(num_state,num_action).to(device)
        self.optimizer = optim.Adam(self.acnet.parameters(),lr=lr)
        self.memory = []

    def update_policy(self):
        R = 0
        actor_loss = 0
        critic_loss = 0
        for r,prob,v in self.memory[::-1]:
            R = r + self.gamma * R
            advantage = R - v
            actor_loss -= torch.log(prob) * advantage
            critic_loss += F.smooth_l1_loss(v,torch.tensor(R).to(device))
        actor_loss /= len(self.memory)
        critic_loss /= len(self.memory)
        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()

    # softmaxの出力が最も大きい行動を選択
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state).to(device)
        action_prob, _ = self.acnet(state_tensor.data)
        action = torch.argmax(action_prob.squeeze().data).item()
        return action

    # カテゴリカル分布からサンプリングして行動を選択
    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state).to(device)
        action_prob, state_value = self.acnet(state_tensor.data)
        action_prob, state_value = action_prob.squeeze(), state_value.squeeze()
        action = Categorical(action_prob).sample().item()
        return action, action_prob[action], state_value

    def add_memory(self, r, prob, v):
        self.memory.append((r, prob, v))

    def reset_memory(self):
        self.memory = []

if __name__ == '__main__':
    # 各種設定
    num_episode = 1200  # 学習エピソード数
    # penalty = 10  # 途中でエピソードが終了したときのペナルティ

    # ログ
    episode_rewards = []
    num_average_epidodes = 10

    env = gym.make('CartPole-v0')
    max_steps = env.spec.max_episode_steps  # エピソードの最大ステップ数

    agent = ActorCriticAgent(env.observation_space.shape[0], env.action_space.n)

    for episode in range(num_episode):
        state = env.reset()  # envからは4次元の連続値の観測が返ってくる
        episode_reward = 0
        for t in range(max_steps):
            action, prob, state_value = agent.get_action(state)  # 行動を選択
            next_state, reward, done, _ = env.step(action)
            #         # もしエピソードの途中で終了してしまったらペナルティを加える
            #         if done and t < max_steps - 1:
            #             reward = - penalty
            episode_reward += reward
            agent.add_memory(reward, prob, state_value)
            state = next_state
            if done:
                agent.update_policy()
                agent.reset_memory()
                break
        episode_rewards.append(episode_reward)
        if episode % 50 == 0:
            print("Episode %d finished | Episode reward %f" % (episode, episode_reward))

    # 累積報酬の移動平均を表示
    moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes) / num_average_epidodes, mode='valid')
    plt.plot(np.arange(len(moving_average)), moving_average)
    plt.title('Actor-Critic: average rewards in %d episodes' % num_average_epidodes)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.show()

    env.close()