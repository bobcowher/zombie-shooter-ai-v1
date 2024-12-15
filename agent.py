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


class Agent():

    def __init__(self, env, dropout, hidden_layer, learning_rate, step_repeat, gamma) -> None:

        self.env = env

        self.step_repeat = step_repeat

        self.gamma = gamma

        observation, info = self.env.reset()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.memory = ReplayBuffer(max_size=500000, input_shape=observation.shape, n_actions=env.action_space.n, device=self.device)

        self.model = ZombieNet(action_dim=env.action_space.n, hidden_dim=hidden_layer, dropout=dropout, observation_shape=observation.shape).to(self.device)

        # self.model.load_the_model()

        self.target_model = ZombieNet(action_dim=env.action_space.n, hidden_dim=hidden_layer, dropout=dropout, observation_shape=observation.shape).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate

        print(f"Initialized agents on device: {self.device}")
        print(f"Memory Size: {asizeof.asizeof(self.memory) / (1024 * 1024 * 1024):2f} Gb")


    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay, min_epsilon):
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)

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
                    # print(state)            
                    q_values = self.model.forward(state.unsqueeze(0).to(self.device))[0]
                    action = torch.argmax(q_values, dim=-1, keepdim=True)

                next_state, reward, done, truncated, info = self.env.step(action=action, repeat=self.step_repeat)

                self.memory.store_transition(state, action, reward, next_state, done)

                state = next_state        

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if self.memory.can_sample(batch_size):
                    states, actions, rewards, next_states, dones = self.memory.sample_buffer(batch_size)

                    # Get Q-values for the current states
                    q_values = self.model(states)

                    # Ensure actions are int64 and reshape for gather
                    actions = actions.unsqueeze(1).long()

                    # print("Q-values shape:", q_values.shape)
                    # print("Actions shape:", actions.shape)

                    # Gather Q-values corresponding to the taken actions
                    qsa_b = q_values.gather(1, actions)

                    # Get target Q-values for next states
                    next_q_values = self.target_model(next_states)

                    # Calculate max Q-value for next states along action dimension
                    max_next_qsa_b = torch.max(next_q_values, dim=1, keepdim=True)[0]



                    # Compute the target using the Bellman equation
                    target_b = rewards.unsqueeze(1) + (~dones.unsqueeze(1)) * self.gamma * max_next_qsa_b

                    # Ensure target_b has the same shape as qsa_b
                    target_b = target_b.expand_as(qsa_b)

                    # print("QSA batch", qsa_b)
                    # print("Target batch", target_b)
                    # time.sleep(5)


                    # Calculate the loss
                    loss = F.smooth_l1_loss(qsa_b, target_b)

                    writer.add_scalar("Loss", loss, total_steps)
                    
                    # Backpropagation and optimization step
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if episode_steps % 4 == 0:
                        soft_update(self.target_model, self.model)
                                


            self.model.save_the_model()
            
            writer.add_scalar('Score', episode_reward, episode)
            writer.add_scalar('Epsilon', epsilon, episode)


            if epsilon > min_epsilon:
                epsilon *= epsilon_decay
            
            episode_time = time.time() - episode_start_time
            

            print(f"Completed episode {episode} with score {episode_reward}")
            print(f"Episode Time: {episode_time:1f} seconds")
            print(f"Episode Steps: {episode_steps}")
            