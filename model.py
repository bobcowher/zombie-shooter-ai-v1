import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sac_utils import *
import time


class ActorCriticBase(nn.Module):

    def __init__(self, action_dim, hidden_size, dropout, observation_shape) -> None:
        super(ActorCriticBase, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        
        # Pooling layer for additional downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        conv_output_size = self.calculate_conv_output(observation_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, action_dim)  # Outputs logits for each action

        self.dropout = dropout

        # Initialize weights
        self.apply(weights_init)

    def calculate_conv_output(self, observation_shape):
        x = torch.zeros(1, *observation_shape)
        x = self.pool(F.relu(self.conv1(x)))  # Pooling after first conv layer
        x = F.relu(self.conv2(x))             # No pooling after second to control size
        x = self.pool(F.relu(self.conv3(x)))  # Pooling after third conv layer
        x = F.relu(self.conv4(x))             # No pooling after second to control size

        return x.view(-1).shape[0]

    def save_the_model(self, weights_filename='models/latest.pt'):
        # Take the default weights filename(latest.pt) and save it
        torch.save(self.state_dict(), weights_filename)


    def load_the_model(self, weights_filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
        except:
            print(f"No weights file available at {weights_filename}")


class Critic(ActorCriticBase):
    def __init__(self, action_dim, hidden_size=256, dropout=0, observation_shape=None):
        super(Critic, self).__init__(action_dim, hidden_size=hidden_size, dropout=dropout, observation_shape=observation_shape)
    

    def forward(self, x):
        # CNN forward pass

        x = x / 255 # Normalize values. 

        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))  # Pooling after third conv layer
        x = F.relu(self.conv4(x)) 
        x = x.view(x.size(0), -1) 

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x
    

class Actor(ActorCriticBase):
    def __init__(self, action_dim, hidden_size=256, dropout=0, observation_shape=None):
        super(Actor, self).__init__(action_dim, hidden_size=hidden_size, dropout=dropout, observation_shape=observation_shape)


    def forward(self, x):
        debug = False

        if debug:
            print("Starting model output --- \n\n\n")
            print("Model Input: ", x)

        x = x / 255 # Normalize values. 

        if debug:
            print("Input after normalization: ", x)
        
        # CNN forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))  # Pooling after third conv layer
        x = F.relu(self.conv4(x)) 
        x = x.view(x.size(0), -1) 

        # Fully connected layers
        if debug:
            print("Conv Output: ", x)
        x = F.relu(self.fc1(x))
        if debug:
            print("X after layer norm: ", x)
        x = F.relu(self.fc2(x))
        if debug:
            print("X after layer 2: ", x)
        x = self.output(x)
        

        # x = F.tanh(self.output(x) / (x.abs().max() + 1e-6))  # Output logits for the action distribution
        
        if debug:
            print("X after output: ", x)
        
        if debug:
            print("Action Probs: ", x)
            print("After softmax", F.softmax(x, dim=-1))
            # time.sleep(1)

        return x
