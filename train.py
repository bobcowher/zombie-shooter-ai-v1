import random
from util import *
from game import ZombieShooter
import time
from agent_sac import Agent


episodes = 3000
max_episode_steps = 2400
total_steps = 0
step_repeat = 4
max_episode_steps = max_episode_steps / step_repeat

batch_size = 64
learning_rate = 0.0001
epsilon = 1
min_epsilon = 0.1
epsilon_decay = 0.99
gamma = 0.99

hidden_size = 256

dropout = 0

# print(observation.shape)

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800  # Visible game window size
WORLD_WIDTH, WORLD_HEIGHT = 1800, 1200  # The size of the larger game world
FPS = 60

env = ZombieShooter(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, world_height=WORLD_HEIGHT, world_width=WORLD_WIDTH, fps=FPS, sound=False, render_mode="rgb")


summary_writer_suffix = f'sac'

agent = Agent(env=env)

    # def __init__(self, num_inputs, num_actions, gamma, tau, alpha, target_update_interval,
    #              automatic_entropy_tuning, hidden_size, learning_rate):

# Training Phase 1

agent.train(2000, max_episode_steps=max_episode_steps, summary_writer_suffix=summary_writer_suffix + "-phase-1",
            batch_size=batch_size, warmup=0)
    

agent.train(2000, max_episode_steps=max_episode_steps * 2, summary_writer_suffix=summary_writer_suffix + "-phase-2",
            batch_size=batch_size, warmup=0)
