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



# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800  # Visible game window size
WORLD_WIDTH, WORLD_HEIGHT = 1800, 1200  # The size of the larger game world
FPS = 60

env = ZombieShooter(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, world_height=WORLD_HEIGHT, world_width=WORLD_WIDTH, fps=FPS, sound=False, render_mode="rgb")

# Game loop

episodes = 10

for episode in range(episodes):

    done = False
    episode_reward = 0
    observation, info = env.reset()


    while not done:

        # Action Mapping
        # [up, down, left, right, switch gun, fire]
        # [W, S, A, D, TAB, SPACE]
        # action = [0,0,0,0,0,0,0]

        action = env.action_space.sample()

        observation, reward, done, truncated, info = env.step(action=action)

        episode_reward += reward

        if reward != 0:
            print("Reward: ", reward)
            print("Observation: ", observation)
            print("Done: ", done)
            print("Info: ", info)
            cv2.imwrite("temp/screen.jpg", observation)
    
    
    

