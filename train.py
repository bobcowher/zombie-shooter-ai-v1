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

env = ZombieShooter(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, world_height=WORLD_HEIGHT, world_width=WORLD_WIDTH, fps=FPS, sound=False, render_mode="rgb")

observation, info = env.reset()

memory = ReplayBuffer(max_size=500000, input_shape=observation.shape, n_actions=env.action_space.n)

# Game loop

episodes = 10
max_episode_steps = 8000
batch_size = 64
learning_rate = 0.0001
epsilon = 0.1
gamma = 0.99

model = Actor(action_dim=env.action_space.n, hidden_dim=256)
target_model = Actor(action_dim=env.action_space.n, hidden_dim=256)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# critic_1 = Critic()

summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_dqn'
writer = SummaryWriter(summary_writer_name)

for episode in range(episodes):

    done = False
    episode_reward = 0
    state, info = env.reset()
    episode_steps = 0

    episode_start_time = time.time()

    while not done and episode_steps < max_episode_steps:

        if random.random() < epsilon or episode_steps < 100:
            action = env.action_space.sample()
        else:
            # print(state)            
            action = model.forward(state.unsqueeze(0))[0]
            action = (action >= 0.5) # Turn probabilities into 0s and 1s

        next_state, reward, done, truncated, info = env.step(action=action)

        memory.store_transition(state, action, reward, next_state, done)

        state = next_state

        episode_reward += reward
        episode_steps += 1

        if memory.can_sample(batch_size):
                states, actions, rewards, next_states, dones = memory.sample_buffer(batch_size)
                qsa_b = model(states).gather(1, actions.long())
                next_qsa_b = target_model(next_states)
                next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                target_b = rewards + ~dones * gamma * next_qsa_b
                print("QSA_B", qsa_b)
                print("Target_B", target_b)
                loss = F.mse_loss(qsa_b, target_b)
                model.zero_grad()
                loss.backward()
                optimizer.step()
        
    
    writer.add_scalar('Score', episode_reward, episode)
    
    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())


    
    episode_time = time.time() - episode_start_time
    
    print(f"Completed episode {episode} with score {episode_reward}")
    print(f"Episode Time: {episode_time:1f} seconds")
    print(f"Episode Steps: {episode_steps}")
    
    
    
    

