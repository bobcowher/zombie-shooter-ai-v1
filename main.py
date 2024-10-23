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

game = ZombieShooter(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, world_height=WORLD_HEIGHT, world_width=WORLD_WIDTH, fps=FPS, sound=False, render_mode="rgb")

# Game loop
while True:


    # Action Mapping
    # [up, down, left, right, switch gun, fire]
    # [W, S, A, D, TAB, SPACE]
    action = [0,0,0,0,0,0,0]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                action[4] = 1
            elif event.key == pygame.K_SPACE:
                action[5] = 1
            elif event.key == pygame.K_ESCAPE:
                action[6] = 1

    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:
        action[0] = 1
    if keys[pygame.K_s]:
        action[1] = 1
    if keys[pygame.K_a]:  # Left
        action[2] = 1
    if keys[pygame.K_d]:  # Right
        action[3] = 1



    observation, reward, done, truncated, info = game.step(action=action)

    if reward != 0:
        print("Reward: ", reward)
        print("Observation: ", observation)
        print("Done: ", done)
        print("Info: ", info)
        cv2.imwrite("temp/screen.jpg", observation)

