"""
Standard Q-Network implementation for Narde game.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """A simple neural network for Q-value estimation"""
    
    def __init__(self, state_size, action_size):
        """
        Initialize the QNetwork.
        
        Args:
            state_size: Dimension of state representation
            action_size: Number of possible actions
        """
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Set the device (MPS, CUDA, or CPU)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                  "cuda" if torch.cuda.is_available() else 
                                  "cpu")
        
        # Initialize the network
        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)
    
    def forward(self, state):
        """
        Forward pass of the network.
        
        Args:
            state: Input state tensor or array
            
        Returns:
            Tensor of Q-values for each action
        """
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        # Ensure proper shape
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            
        # Ensure state is on the right device
        state = state.to(self.device)
        
        return self.network(state) 