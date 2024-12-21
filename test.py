import datetime
import pygame
import sys
import math
from assets import Zombie, Player
from bullet import SingleBullet
import random
from util import *
from game import ZombieShooter
import cv2
import os
import time
from buffer import ReplayBuffer
from model import ZombieNet
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800  # Visible game window size
WORLD_WIDTH, WORLD_HEIGHT = 1800, 1200  # The size of the larger game world
FPS = 60

env = ZombieShooter(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, world_height=WORLD_HEIGHT, world_width=WORLD_WIDTH, fps=FPS, sound=False, render_mode="human")

observation, info = env.reset()


# Game loop

episodes = 1
max_episode_steps = 1200
batch_size = 64
learning_rate = 0.0001
epsilon = 0.05
min_epsilon = 0.1
epsilon_decay = 0.99
gamma = 0.99
hidden_layer = 1024


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

memory = ReplayBuffer(max_size=500000, input_shape=observation.shape, n_actions=env.action_space.n, device=device)

model1 = ZombieNet(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=observation.shape).to(device)
model2 = ZombieNet(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=observation.shape).to(device)

model1.load_the_model(filename='models/dqn1.pt')
model2.load_the_model(filename='models/dqn2.pt')

model1.eval()
model2.eval()

# critic_1 = Critic()

step_repeat = 4

for episode in range(episodes):

    done = False
    episode_reward = 0
    state, info = env.reset()
    episode_steps = 0

    episode_start_time = time.time()

    while not done and episode_steps < max_episode_steps:

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            # print(state)            
            model1_q_values = model1.forward(state.unsqueeze(0).to(device))[0]
            model2_q_values = model2.forward(state.unsqueeze(0).to(device))[0]
            q_values = torch.min(model1_q_values, model2_q_values)

            action = torch.argmax(q_values, dim=-1, keepdim=True)

        next_state, reward, done, truncated, info = env.step(action=action, repeat=step_repeat)

        state = next_state

        episode_reward += reward
        episode_steps += 1



    
    episode_time = time.time() - episode_start_time
    
    print(f"Completed episode {episode} with score {episode_reward}")
    print(f"Episode Time: {episode_time:1f} seconds")
    print(f"Episode Steps: {episode_steps}")
    

    
    
    

