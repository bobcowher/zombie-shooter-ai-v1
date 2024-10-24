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
from model import Actor, Critic
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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

memory = ReplayBuffer(max_size=500000, input_shape=observation.shape, n_actions=env.action_space.n, device=device)

model = Actor(action_dim=env.action_space.n, hidden_dim=256).to(device)

model.load_the_model()

target_model = Actor(action_dim=env.action_space.n, hidden_dim=256).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# critic_1 = Critic()

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
            action = model.forward(state.unsqueeze(0).to(device))[0]
            action = (action >= 0.5) # Turn probabilities into 0s and 1s

        next_state, reward, done, truncated, info = env.step(action=action)

        state = next_state

        episode_reward += reward
        episode_steps += 1



    
    episode_time = time.time() - episode_start_time
    
    print(f"Completed episode {episode} with score {episode_reward}")
    print(f"Episode Time: {episode_time:1f} seconds")
    print(f"Episode Steps: {episode_steps}")
    

    
    
    

