import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ZombieNet(nn.Module):
    def __init__(self, action_dim, hidden_dim=256, dropout=0, observation_shape=None):
        super(ZombieNet, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        
        conv_output_size = self.calculate_conv_output(observation_shape)

        print("conv_output_size: ", conv_output_size)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_dim * 2)  # Adjust based on input size
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

        self.dropout = dropout

        # Initialize weights
        self.apply(self.weights_init)
    
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

        q_values = self.output(x)
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

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            # Initialize Conv2d layers with Kaiming (He) initialization
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Initialize Linear layers with Xavier (Glorot) initialization
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# Helper Function
def soft_update(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)