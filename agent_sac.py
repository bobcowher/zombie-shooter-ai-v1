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


class Agent(object):
    def __init__(self, env, hidden_size=512, gamma=0.99, 
                 tau=0.005, alpha=0.1, target_update_interval=1, 
                 learning_rate=0.0003, step_repeat=4):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.step_repeat = step_repeat

        self.env = env

        observation, info = self.env.reset()

        self.memory = ReplayBuffer(max_size=500000, input_shape=observation.shape, n_actions=env.action_space.n, device=self.device)


        # Q-network now outputs values for each discrete action
        self.critic = Critic(observation_shape=observation.shape, 
                             action_dim=env.action_space.n, 
                             hidden_size=hidden_size).to(device=self.device)
        
        self.critic_optim = AdamW(self.critic.parameters(), lr=learning_rate)

        self.critic_target = Critic(observation_shape=observation.shape, 
                                    action_dim=env.action_space.n, 
                                    hidden_size=hidden_size).to(device=self.device)

        hard_update(self.critic_target, self.critic)

        # Policy network for discrete actions: outputs logits for each action
        self.policy = Actor(observation_shape=observation.shape, 
                            action_dim=env.action_space.n, 
                            hidden_size=hidden_size).to(self.device)
        
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        logits = self.policy(state)
        probs = F.softmax(logits, dim=-1)
        if evaluate:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
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
            # Epsilon-greedy action selection
            logits = self.policy(state)
            action = torch.argmax(logits, dim=-1, keepdim=True)

            # Environment step
            next_state, reward, done, truncated, info = self.env.step(action=action, repeat=self.step_repeat)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

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


    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay, min_epsilon):
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

            while not done and episode_steps < max_episode_steps:
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    logits = self.policy(state)
                    action = torch.argmax(logits, dim=-1, keepdim=True)

                # Environment step
                next_state, reward, done, truncated, info = self.env.step(action=action, repeat=self.step_repeat)
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

                # Store the transition in memory
                self.memory.store_transition(state, action.item(), reward, next_state, done)
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

                # Update target networks at intervals
                if total_steps % self.target_update_interval == 0:
                    soft_update(self.critic_target, self.critic, self.tau)

            # Save model after each episode
            self.policy.save_the_model()
            
            # Log episode results to TensorBoard
            writer.add_scalar('Score', episode_reward, episode)
            writer.add_scalar('Epsilon', epsilon, episode)
            
            # Update epsilon (exploration decay)
            if epsilon > min_epsilon:
                epsilon *= epsilon_decay
            
            # Print episode summary
            episode_time = time.time() - episode_start_time
            print(f"Completed episode {episode} with score {episode_reward}")
            print(f"Episode Time: {episode_time:.1f} seconds")
            print(f"Episode Steps: {episode_steps}")



    def update_parameters(self, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample_buffer(batch_size=batch_size)

        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = action_batch.unsqueeze(1)
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)


        with torch.no_grad():
            next_q_values = self.critic_target(next_state_batch)
            next_q_value = torch.max(next_q_values, dim=1, keepdim=True)[0]  # Shape: [batch_size, 1]
            q_target = reward_batch + ~done_batch * self.gamma * next_q_value  # Shape: [batch_size, 1]

        # Q-values for current state-action pairs
        # print(action_batch)
        q_values = self.critic(state_batch).gather(1, action_batch)
        # print("Q_Values: ", q_values)
        # print("Q_Target", q_target)
        qf_loss = F.mse_loss(q_values, q_target)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Update policy using categorical cross-entropy loss
        logits = self.policy(state_batch)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        qf_pi = self.critic(state_batch).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        policy_loss = (self.alpha * log_probs - qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Soft update target network
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf_loss.item(), policy_loss.item()