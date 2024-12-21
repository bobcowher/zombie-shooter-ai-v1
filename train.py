import random
from util import *
from game import ZombieShooter
import time
from agent import Agent


episodes = 10000
max_episode_steps = 10000
total_steps = 0
step_repeat = 4
max_episode_steps = max_episode_steps / step_repeat

batch_size = 64
learning_rate = 0.0001
epsilon = 1
min_epsilon = 0.1
epsilon_decay = 0.995
gamma = 0.99

hidden_layer = 1024

dropout = 0.2

# print(observation.shape)

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800  # Visible game window size
WORLD_WIDTH, WORLD_HEIGHT = 1800, 1200  # The size of the larger game world
FPS = 60

env = ZombieShooter(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, world_height=WORLD_HEIGHT, world_width=WORLD_WIDTH, fps=FPS, sound=False, render_mode="rgb")


summary_writer_suffix = f'dqn_lr={learning_rate}_hl={hidden_layer}_mse_loss_bs={batch_size}_dropout={dropout}_double_dqn'

agent = Agent(env, dropout=0.2, hidden_layer=hidden_layer,
              learning_rate=learning_rate, step_repeat=step_repeat,
              gamma=gamma)


# Training Phase 1

agent.train(episodes=episodes, max_episode_steps=max_episode_steps, summary_writer_suffix=summary_writer_suffix + "-phase-1",
            batch_size=batch_size, epsilon=epsilon, epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon)
    

