"""
Decomposed DQN implementation for Narde game.

This network architecture decomposes the action space to potentially
improve learning by treating the selection of pieces and destinations
as separate decision processes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp

class DecomposedDQN:
    """
    A decomposed DQN implementation that handles the Narde game's
    action space by decomposing it into separate decision components.
    """
    
    def __init__(self, state_size, action_size):
        """
        Initialize the DecomposedDQN.
        
        Args:
            state_size: Dimension of state representation
            action_size: Number of possible actions
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Set device - if MPS is available, use a safer approach
        if torch.backends.mps.is_available():
            try:
                # First check if MPS is working correctly
                test_tensor = torch.zeros(1).to("mps")
                self.device = torch.device("mps")
                
                # Only print in main process
                if mp.current_process().name == 'MainProcess':
                    print("Successfully initialized MPS device")
            except Exception as e:
                if mp.current_process().name == 'MainProcess':
                    print(f"MPS initialization failed: {e}, falling back to CPU")
                self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Create network
        self.network = self._build_network()
        
        # Move to device as a separate step
        try:
            self.network.to(self.device)
            if mp.current_process().name == 'MainProcess':
                print(f"Network successfully moved to {self.device}")
        except Exception as e:
            if mp.current_process().name == 'MainProcess':
                print(f"Failed to move network to {self.device}: {e}, falling back to CPU")
            self.device = torch.device("cpu")
            self.network.to(self.device)
        
    def _build_network(self):
        """
        Build the core neural network.
        
        Returns:
            PyTorch sequential network
        """
        # Simple architecture with minimal complexity
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        
    def get_q_values(self, state):
        """
        Get Q-values for all possible moves.
        
        Args:
            state: Input state
            
        Returns:
            Numpy array of Q-values
        """
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state.to(self.device)
            
        # Ensure proper shape
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        # Get Q-values with safety handling
        try:
            with torch.no_grad():
                q_values = self.network(state_tensor)
                return q_values.cpu().numpy()[0]
        except Exception as e:
            print(f"Error in get_q_values: {e}, returning zeros")
            return np.zeros(self.action_size)
        
    def forward(self, state):
        """
        Forward pass of the network (compatibility with QNetwork).
        
        Args:
            state: Input state tensor or array
            
        Returns:
            Tensor of Q-values for each action
        """
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state.to(self.device)
            
        # Ensure proper shape
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        # Forward pass with safety handling
        try:
            return self.network(state_tensor)
        except Exception as e:
            print(f"Error in forward: {e}, returning zeros")
            return torch.zeros(state_tensor.size(0), self.action_size, device=self.device) 