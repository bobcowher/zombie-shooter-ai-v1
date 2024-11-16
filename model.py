import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sac_utils import *

class Critic(nn.Module):
    def __init__(self, action_dim, hidden_size=256, dropout=0, observation_shape=None):
        super(Critic, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        
        conv_output_size = self.calculate_conv_output(observation_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, action_dim)  # Outputs Q-values for each action

        self.dropout = dropout

        # Initialize weights
        self.apply(weights_init)

    def calculate_conv_output(self, observation_shape):
        x = np.zeros(observation_shape, dtype=np.float32)
        x = torch.tensor(x).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x.shape[1]
    

    def forward(self, x):
        # CNN forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the feature map

        # Fully connected layers
        x = F.relu(self.fc1(x))
        if self.dropout > 0:
            x = F.dropout(x, self.dropout)
        x = F.relu(self.fc2(x))
        if self.dropout > 0:
            x = F.dropout(x, self.dropout)

        q_values = self.output(x)  # Output Q-values for each action
        return q_values
    
    
    def save_the_model(self, weights_filename='models/latest.pt'):
        # Take the default weights filename(latest.pt) and save it
        torch.save(self.state_dict(), weights_filename)


    def load_the_model(self, weights_filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
        except:
            print(f"No weights file available at {weights_filename}")


class Actor(nn.Module):
    def __init__(self, action_dim, hidden_size=256, dropout=0, observation_shape=None):
        super(Actor, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        
        conv_output_size = self.calculate_conv_output(observation_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, action_dim)  # Outputs logits for each action

        self.dropout = dropout

        # Initialize weights
        self.apply(weights_init)

    def calculate_conv_output(self, observation_shape):
        x = np.zeros(observation_shape, dtype=np.float32)
        x = torch.tensor(x).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x.shape[1]

    def forward(self, x):
        # CNN forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the feature map

        # Fully connected layers
        x = F.relu(self.fc1(x))
        if self.dropout > 0:
            x = F.dropout(x, self.dropout)
        x = F.relu(self.fc2(x))
        if self.dropout > 0:
            x = F.dropout(x, self.dropout)

        logits = self.output(x)  # Output logits for the action distribution
        return logits

    def save_the_model(self, weights_filename='models/latest.pt'):
        # Take the default weights filename(latest.pt) and save it
        torch.save(self.state_dict(), weights_filename)


    def load_the_model(self, weights_filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
        except:
            print(f"No weights file available at {weights_filename}")