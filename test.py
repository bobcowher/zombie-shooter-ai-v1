from util import *
from game import ZombieShooter
from agent_sac import Agent

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
hidden_size = 512

agent = Agent(env=env, hidden_size=hidden_size)

agent.test(max_episode_steps=1200)

    
    
    

