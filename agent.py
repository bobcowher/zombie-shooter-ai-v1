from assets import Zombie, Player
from buffer import ReplayBuffer
from model import ZombieNet, soft_update, hard_update
import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import random
from pympler import asizeof
import os

class Agent():

    def __init__(self, env, dropout, hidden_layer, learning_rate, step_repeat, gamma) -> None:

        self.env = env

        self.step_repeat = step_repeat

        self.gamma = gamma

        observation, info = self.env.reset()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.memory = ReplayBuffer(max_size=500000, input_shape=observation.shape, n_actions=env.action_space.n, device=self.device)

        self.model_1 = ZombieNet(action_dim=env.action_space.n, hidden_dim=hidden_layer, dropout=dropout, observation_shape=observation.shape).to(self.device)
        self.model_2 = ZombieNet(action_dim=env.action_space.n, hidden_dim=hidden_layer, dropout=dropout, observation_shape=observation.shape).to(self.device)

        self.target_model_1 = ZombieNet(action_dim=env.action_space.n, hidden_dim=hidden_layer, dropout=dropout, observation_shape=observation.shape).to(self.device)
        self.target_model_2 = ZombieNet(action_dim=env.action_space.n, hidden_dim=hidden_layer, dropout=dropout, observation_shape=observation.shape).to(self.device)

        # Initialize target networks with model parameters
        self.target_model_1.load_state_dict(self.model_1.state_dict())
        self.target_model_2.load_state_dict(self.model_2.state_dict())

        self.optimizer_1 = optim.Adam(self.model_1.parameters(), lr=learning_rate)
        self.optimizer_2 = optim.Adam(self.model_2.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate

        print(f"Initialized agents on device: {self.device}")
        print(f"Memory Size: {asizeof.asizeof(self.memory) / (1024 * 1024 * 1024):2f} Gb")

    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay, min_epsilon):
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)

        if not os.path.exists('models'):
            os.makedirs('models')

        total_steps = 0

        for episode in range(episodes):

            done = False
            episode_reward = 0
            state, info = self.env.reset()
            episode_steps = 0

            episode_start_time = time.time()

            while not done and episode_steps < max_episode_steps:

                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    q_values_1 = self.model_1.forward(state.unsqueeze(0).to(self.device))[0]
                    q_values_2 = self.model_2.forward(state.unsqueeze(0).to(self.device))[0]
                    q_values = torch.min(q_values_1, q_values_2)
                    action = torch.argmax(q_values, dim=-1).item()

                next_state, reward, done, truncated, info = self.env.step(action=action, repeat=self.step_repeat)

                self.memory.store_transition(state, action, reward, next_state, done)

                state = next_state        

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if self.memory.can_sample(batch_size):
                    states, actions, rewards, next_states, dones = self.memory.sample_buffer(batch_size)

                    dones = dones.unsqueeze(1).float()

                    # Current Q-values from both models
                    q_values_1 = self.model_1(states)
                    q_values_2 = self.model_2(states)
                    actions = actions.unsqueeze(1).long()
                    qsa_b_1 = q_values_1.gather(1, actions)
                    qsa_b_2 = q_values_2.gather(1, actions)

                    # Action selection using the main models
                    next_actions_1 = torch.argmax(self.model_1(next_states), dim=1, keepdim=True)
                    next_actions_2 = torch.argmax(self.model_2(next_states), dim=1, keepdim=True)

                    # Q-value evaluation using the target models
                    next_q_values_1 = self.target_model_1(next_states).gather(1, next_actions_1)
                    next_q_values_2 = self.target_model_2(next_states).gather(1, next_actions_2)

                    # Take the minimum of the next Q-values
                    next_q_values = torch.min(next_q_values_1, next_q_values_2)

                    # Compute the target using Double DQN with minimization
                    target_b = rewards.unsqueeze(1) + (1 - dones) * self.gamma * next_q_values

                    # Calculate the loss for both models
                    loss_1 = F.smooth_l1_loss(qsa_b_1, target_b.detach())
                    loss_2 = F.smooth_l1_loss(qsa_b_2, target_b.detach())

                    writer.add_scalar("Loss/Model_1", loss_1.item(), total_steps)
                    writer.add_scalar("Loss/Model_2", loss_2.item(), total_steps)

                    # Backpropagation and optimization step for both models
                    self.model_1.zero_grad()
                    loss_1.backward()
                    self.optimizer_1.step()

                    self.model_2.zero_grad()
                    loss_2.backward()
                    self.optimizer_2.step()

                    # Update the target models periodically
                    if episode_steps % 4 == 0:
                        soft_update(self.target_model_1, self.model_1)
                        soft_update(self.target_model_2, self.model_2)

            self.model_1.save_the_model(filename='models/dqn1.pt')
            self.model_2.save_the_model(filename='models/dqn2.pt')

            writer.add_scalar('Score', episode_reward, episode)
            writer.add_scalar('Epsilon', epsilon, episode)

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay

            episode_time = time.time() - episode_start_time

            print(f"Completed episode {episode} with score {episode_reward}")
            print(f"Episode Time: {episode_time:1f} seconds")
            print(f"Episode Steps: {episode_steps}")
