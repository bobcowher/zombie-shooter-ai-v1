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

game = ZombieShooter(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, world_height=WORLD_HEIGHT, world_width=WORLD_WIDTH, fps=FPS, sound=False, render_mode="human")

# Game loop
while True:
    observation, reward, done, truncated, info = game.step(action=[1,0,0,0,0,0])

    if reward != 0:
        print("Reward: ", reward)
        print("Observation: ", observation)
        print("Done: ", done)
        print("Info: ", info)
        # cv2.imwrite("temp/screen.jpg", observation)

