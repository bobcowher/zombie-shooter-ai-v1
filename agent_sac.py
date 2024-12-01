import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from sac_utils import *
from model import *
import time
import datetime
import random
from buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from pympler import asizeof

class Agent(object):
    def __init__(self, env, hidden_size=512, gamma=0.99, 
                 tau=0.005, alpha=0.1, target_update_interval=1, 
                 learning_rate=0.0001, step_repeat=4):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.step_repeat = step_repeat

        self.env = env

        observation, info = self.env.reset()

        self.memory = ReplayBuffer(max_size=500000, input_shape=observation.shape, n_actions=env.action_space.n, device=self.device)

        print(f"Initialized agents on device: {self.device}")
        print(f"Memory Size: {asizeof.asizeof(self.memory) / (1024 * 1024 * 1024):2f} Gb")


        # Q-network now outputs values for each discrete action
        self.critic1 = Critic(observation_shape=observation.shape, 
                             action_dim=env.action_space.n, 
                             hidden_size=hidden_size).to(device=self.device)
        
        self.critic1_optim = AdamW(self.critic1.parameters(), lr=learning_rate * 3, weight_decay=0.0001)

        self.critic1_target = Critic(observation_shape=observation.shape, 
                                    action_dim=env.action_space.n, 
                                    hidden_size=hidden_size).to(device=self.device)

        hard_update(self.critic1_target, self.critic1)

        self.critic2 = Critic(observation_shape=observation.shape, 
                             action_dim=env.action_space.n, 
                             hidden_size=hidden_size).to(device=self.device)
        
        self.critic2_optim = AdamW(self.critic2.parameters(), lr=learning_rate * 3, weight_decay=0.0001)

        self.critic2_target = Critic(observation_shape=observation.shape, 
                                    action_dim=env.action_space.n, 
                                    hidden_size=hidden_size).to(device=self.device)

        hard_update(self.critic2_target, self.critic2)

        # Policy network for discrete actions: outputs logits for each action
        self.policy = Actor(observation_shape=observation.shape, 
                            action_dim=env.action_space.n, 
                            hidden_size=hidden_size).to(self.device)

        # self.policy.load_the_model()
        
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)


    def select_action(self, state, evaluate=False):
        logits = self.policy(state)
        probs = F.softmax(logits, dim=-1)
        # print("Logits: ", logits)
        # print("Probs: ", probs)

        if evaluate:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        # print("Actions selected: ", action)
        # time.sleep(1)
        return action.item()

    def test(self, max_episode_steps):

        self.policy.load_the_model()

        total_steps = 0

        done = False
        episode_reward = 0
        state, info = self.env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        episode_steps = 0
        episode_start_time = time.time()

        while not done and episode_steps < max_episode_steps:

            action = self.select_action(state, evaluate=False)

            # Environment step
            next_state, reward, done, truncated, info = self.env.step(action=action, repeat=self.step_repeat)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

            # print("Reward: ", reward)

            # Store the transition in memory
            state = next_state  # Update current state

            # Accumulate episode reward and steps
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

        
        # Print episode summary
        episode_time = time.time() - episode_start_time
        print(f"Completed episode with score {episode_reward}")
        print(f"Episode Time: {episode_time:.1f} seconds")
        print(f"Episode Steps: {episode_steps}")


    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, warmup=10, epsilon=1, min_epsilon=0.1, epsilon_decay=0.99):
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)

        total_steps = 0

        for episode in range(episodes):
            done = False
            episode_reward = 0
            state, info = self.env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            episode_steps = 0
            episode_start_time = time.time()

            if epsilon > min_epsilon:
                epsilon = epsilon * epsilon_decay

            while not done and episode_steps < max_episode_steps:

                if epsilon > random.random():
                    action = self.env.action_space.sample()
                else:
                    action = self.select_action(state=state)


                # Environment step
                next_state, reward, done, truncated, info = self.env.step(action=action, repeat=self.step_repeat)
                # print("Next State: ", next_state)
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                # print("Next State - FloatTensor Unsqueezed: ", next_state)

                # Store the transition in memory
                self.memory.store_transition(state, action, reward, next_state, done)
                state = next_state  # Update current state

                # Accumulate episode reward and steps
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # Sample a batch and update parameters if enough samples are available
                if self.memory.can_sample(batch_size):
                    qf1_loss, policy_loss = self.update_parameters(batch_size=batch_size, updates=total_steps)
                    writer.add_scalar('Critic Loss', qf1_loss, total_steps)
                    writer.add_scalar('Actor Loss', policy_loss, total_steps)
                    writer.add_scalar('Epsilon: ', epsilon, total_steps)


            # Save model after each episode
            self.policy.save_the_model(weights_filename='models/policy.pt')
            self.critic1.save_the_model(weights_filename='models/critic1.pt')
            self.critic2.save_the_model(weights_filename='models/critic2.pt')

            
            # Log episode results to TensorBoard
            writer.add_scalar('Score', episode_reward, episode)
            
            # Print episode summary
            episode_time = time.time() - episode_start_time
            print(f"Completed episode {episode} with score {episode_reward}")
            print(f"Episode Time: {episode_time:.1f} seconds")
            print(f"Episode Steps: {episode_steps}")



    def update_parameters(self, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample_buffer(batch_size=batch_size)

        action_batch = action_batch.unsqueeze(1)
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1).float()

        # Compute target Q-values using double Q-network
        with torch.no_grad():

            next_logits = self.policy(next_state_batch)  # Policy outputs logits
            next_probs = F.softmax(next_logits, dim=-1)
            next_dist = torch.distributions.Categorical(next_probs)
            next_actions = next_dist.sample()  # Sample next actions

            next_q1_values = self.critic1_target(next_state_batch)
            next_q2_values = self.critic2_target(next_state_batch)
            next_q_values = torch.min(next_q1_values, next_q2_values)  # Take the minimum Q-value
            next_q_value = next_q_values.gather(1, next_actions.unsqueeze(-1))

            # if updates % 1000 == 0:
            #     print("Next Q1 Values: ", next_q1_values)
            #     print("Next Q2 Values: ", next_q2_values)
            #     print("Next Q Values: ", next_q_value)

            # Add entropy term to the target
            next_log_probs = next_dist.log_prob(next_actions).unsqueeze(-1)  # Shape: [batch_size, 1]
            q_target = reward_batch + (1 - done_batch) * self.gamma * (next_q_value - self.alpha * next_log_probs)

            # print("Q Target: ", q_target)

        # Compute Q-value predictions for both critics
        q1_values = self.critic1(state_batch).gather(1, action_batch)
        q2_values = self.critic2(state_batch).gather(1, action_batch)

        # Compute losses for both critics
        # print("Q1 Values: ", q1_values)
        # print("Q2 Values: ", q2_values)
        # print("Q Target: ", q_target)

        qf1_loss = F.mse_loss(q1_values, q_target)
        qf2_loss = F.mse_loss(q2_values, q_target)

        # Optimize both critics
        self.critic1_optim.zero_grad()
        qf1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        qf2_loss.backward()
        self.critic2_optim.step()

        # Update target networks
        soft_update(self.critic1_target, self.critic1, self.tau)
        soft_update(self.critic2_target, self.critic2, self.tau)

        logits = self.policy(state_batch)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        # print("Actions: ", actions)
        # time.sleep(1)
        log_probs = dist.log_prob(actions)

        # Use minimum Q-value for policy gradient
        qf_pi = torch.min(
            self.critic1(state_batch).gather(1, actions.unsqueeze(-1)).squeeze(-1),
            self.critic2(state_batch).gather(1, actions.unsqueeze(-1)).squeeze(-1),
        )

        policy_loss = (self.alpha * log_probs - qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return qf1_loss.item(), policy_loss.item()