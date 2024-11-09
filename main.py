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
import numpy as np
import torch


# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800  # Visible game window size
WORLD_WIDTH, WORLD_HEIGHT = 1800, 1200  # The size of the larger game world
FPS = 60

game = ZombieShooter(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, world_height=WORLD_HEIGHT, world_width=WORLD_WIDTH, fps=FPS, sound=True, render_mode="human")

# Game loop
while True:


    # Action Mapping
    # [up, down, left, right, switch gun, fire]
    # [W, S, A, D, TAB, SPACE]
    # action = [0,0,0,0,0,0,0]
    action = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                action = 5
            elif event.key == pygame.K_SPACE:
                action = 6
            elif event.key == pygame.K_ESCAPE:
                action = 7

    keys = pygame.key.get_pressed()

    if action == 0:
        if keys[pygame.K_w]:
            action = 1
        if keys[pygame.K_s]:
            action = 2
        if keys[pygame.K_a]:  # Left
            action = 3
        if keys[pygame.K_d]:  # Right
            action = 4



    observation, reward, done, truncated, info = game.step(action=action)

    if reward != 0:
        print("Reward: ", reward)
        print("Observation: ", observation)
        print("Done: ", done)
        print("Info: ", info)

        img_array = torch.clip(observation.squeeze(0), 0, 255).numpy().astype(np.uint8)

        success = cv2.imwrite("temp/screen.jpg", img_array)

        if not success:
            print("Failed to write the image.")
        else:
            print("Image written successfully!")

        memory_size_mb = img_array.nbytes / (1024 * 1024)

        buffer_size = 200000

        print(f"Memory: {memory_size_mb} Mb")

        expected_buffer_size_gb = (memory_size_mb * buffer_size) / 1024

        print(f"Expected Buffer Size: {expected_buffer_size_gb} Gb")
