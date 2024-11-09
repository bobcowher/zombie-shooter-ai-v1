import datetime
from assets import Zombie, Player
from bullet import SingleBullet
import random
from util import *
from game import ZombieShooter
import time
from buffer import ReplayBuffer
from model import ZombieNet, soft_update, hard_update
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pympler import asizeof


# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800  # Visible game window size
WORLD_WIDTH, WORLD_HEIGHT = 1800, 1200  # The size of the larger game world
FPS = 60

env = ZombieShooter(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, world_height=WORLD_HEIGHT, world_width=WORLD_WIDTH, fps=FPS, sound=False, render_mode="rgb")

observation, info = env.reset()


# Game loop

episodes = 3000
max_episode_steps = 2400
total_steps = 0
step_repeat = 4
max_episode_steps = max_episode_steps / step_repeat

batch_size = 32
learning_rate = 0.0001
epsilon = 0.4
min_epsilon = 0.1
epsilon_decay = 0.99
gamma = 0.99

hidden_layer = 512

dropout = 0

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f"Starting training with device: {device}")

print(observation.shape)

memory = ReplayBuffer(max_size=500000, input_shape=observation.shape, n_actions=env.action_space.n, device=device)

model = ZombieNet(action_dim=env.action_space.n, hidden_dim=hidden_layer, dropout=dropout, observation_shape=observation.shape).to(device)

# model.load_the_model()

target_model = ZombieNet(action_dim=env.action_space.n, hidden_dim=hidden_layer, dropout=dropout, observation_shape=observation.shape).to(device)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# critic_1 = Critic()

summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_dqn_lr={learning_rate}_ed={epsilon_decay}_hl={hidden_layer}_l1_loss_bs={batch_size}_dropout={dropout}'
writer = SummaryWriter(summary_writer_name)



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
            q_values = model.forward(state.unsqueeze(0).to(device))[0]
            action = torch.argmax(q_values, dim=-1, keepdim=True)

        next_state, reward, done, truncated, info = env.step(action=action, repeat=step_repeat)

        memory.store_transition(state, action, reward, next_state, done)

        state = next_state        

        episode_reward += reward
        episode_steps += 1
        total_steps += 1

        if memory.can_sample(batch_size):
            states, actions, rewards, next_states, dones = memory.sample_buffer(batch_size)

            # Get Q-values for the current states
            q_values = model(states)

            # Ensure actions are int64 and reshape for gather
            actions = actions.long()

            # Gather Q-values corresponding to the taken actions
            qsa_b = q_values.gather(1, actions)

            # Get target Q-values for next states
            next_q_values = target_model(next_states)

            # Calculate max Q-value for next states along action dimension
            max_next_qsa_b = torch.max(next_q_values, dim=1, keepdim=True)[0]

            # Compute the target using the Bellman equation
            target_b = rewards.unsqueeze(1) + (~dones.unsqueeze(1)) * gamma * max_next_qsa_b

            # Ensure target_b has the same shape as qsa_b
            target_b = target_b.expand_as(qsa_b)

            # Calculate the loss
            loss = F.smooth_l1_loss(qsa_b, target_b)

            writer.add_scalar("Loss", loss, total_steps)
            
            # Backpropagation and optimization step
            model.zero_grad()
            loss.backward()
            optimizer.step()
        

        if episode_steps % 100 == 0:
            soft_update(target_model, model)


    model.save_the_model()
    
    writer.add_scalar('Score', episode_reward, episode)
    writer.add_scalar('Epsilon', epsilon, episode)


    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
            


    
    episode_time = time.time() - episode_start_time
    
    # objgraph.show_most_common_types(limit=10)

    # objgraph.show_growth(limit=10)



    print(f"Completed episode {episode} with score {episode_reward}")
    print(f"Episode Time: {episode_time:1f} seconds")
    print(f"Episode Steps: {episode_steps}")
    print(f"Bullets size: {len(env.bullets)}")
    print(f"Memory Size: {asizeof.asizeof(memory) / (1024 * 1024 * 1024):2f} Gb")
    

model.save_the_model()
    
    
    

