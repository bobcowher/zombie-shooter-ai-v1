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


for episode in range(episodes):

    done = False
    episode_reward = 0
    state, info = env.reset()
    episode_steps = 0

    episode_start_time = time.time()

    while not done and episode_steps < max_episode_steps:

        # Action Mapping
        # [up, down, left, right, switch gun, fire]
        # [W, S, A, D, TAB, SPACE]
        # action = [0,0,0,0,0,0,0]

        action = env.action_space.sample()

        next_state, reward, done, truncated, info = env.step(action=action)

        memory.store_transition(state, action, reward, next_state, done)

        episode_reward += reward
        episode_steps += 1


        # if episode_steps % 100 == 0:
        #     cv2.imwrite("temp/screen.jpg", observation)
        #     hundred_step_timer = time.time() - last_check_time
        #     print(f"On step {episode_steps} of episode {episode}. Time Taken: {hundred_step_timer}")
        #     last_check_time = time.time()

        # if reward != 0:
        #     print("Reward: ", reward)
        #     print("Observation: ", observation)
        #     print("Done: ", done)
        #     print("Info: ", info)
        #     cv2.imwrite("temp/screen.jpg", observation)
    
    episode_time = time.time() - episode_start_time
    
    print(f"Completed episode {episode} with score {episode_reward}")
    print(f"Episode Time: {episode_time:1f} seconds")
    print(f"Episode Steps: {episode_steps}")
    
    
    
    

