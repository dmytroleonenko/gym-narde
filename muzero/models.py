import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RepresentationNetwork(nn.Module):
    """
    Encodes a game observation into a latent state representation.
    """
    def __init__(self, input_dim, hidden_dim):
        super(RepresentationNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, observation):
        # observation: [batch, input_dim]
        return self.fc(observation)


class DynamicsNetwork(nn.Module):
    """
    Predicts the next latent state and the immediate reward given a
    current latent state and an action.
    """
    def __init__(self, hidden_dim, action_dim):
        super(DynamicsNetwork, self).__init__()
        # Combine hidden state and action (one-hot)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Reward head: predict immediate reward from the next hidden state
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, hidden, action_onehot):
        # hidden: [batch, hidden_dim], action_onehot: [batch, action_dim]
        x = torch.cat([hidden, action_onehot], dim=1)
        next_hidden = self.fc(x)
        reward = self.reward_head(next_hidden)
        return next_hidden, reward


class PredictionNetwork(nn.Module):
    """
    Predicts the policy and value for a given latent state.
    """
    def __init__(self, hidden_dim, action_dim):
        super(PredictionNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Value head: predicts the expected return
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Value normalized to [-1, 1]
        )
        
        # Policy head: predicts action probabilities
        self.policy_head = nn.Linear(128, action_dim)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, hidden):
        # hidden: [batch, hidden_dim]
        x = self.fc(hidden)
        value = self.value_head(x)
        policy_logits = self.policy_head(x)
        return value, policy_logits


class MuZeroNetwork(nn.Module):
    """
    Full MuZero network combining representation, dynamics, and prediction networks.
    """
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(MuZeroNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Networks
        self.representation_network = RepresentationNetwork(input_dim, hidden_dim)
        self.dynamics_network = DynamicsNetwork(hidden_dim, action_dim)
        self.prediction_network = PredictionNetwork(hidden_dim, action_dim)
        
    def initial_inference(self, observation):
        """
        Initial inference for the root node in MCTS.
        """
        hidden = self.representation_network(observation)
        value, policy_logits = self.prediction_network(hidden)
        return hidden, value, policy_logits
    
    def recurrent_inference(self, hidden, action):
        """
        Recurrent inference for non-root nodes in MCTS.
        """
        # Convert action to one-hot encoding
        # If action is already a tensor, use it directly
        if isinstance(action, torch.Tensor):
            # Make sure action is 2D (batch_size, 1)
            if action.dim() == 1:
                action = action.unsqueeze(1)
            
            # Create one-hot tensor of appropriate shape
            batch_size = hidden.size(0)
            action_onehot = torch.zeros(batch_size, self.action_dim, device=hidden.device)
            
            # Handle different action shapes
            if action.size(1) == 1:
                # Single index per batch item
                for i in range(batch_size):
                    action_onehot[i, action[i, 0].long()] = 1.0
            else:
                # Action is already in one-hot format
                action_onehot = action
        else:
            # Convert single integer to one-hot
            batch_size = hidden.size(0)
            action_onehot = torch.zeros(batch_size, self.action_dim, device=hidden.device)
            action_onehot[:, action] = 1.0
        
        next_hidden, reward = self.dynamics_network(hidden, action_onehot)
        value, policy_logits = self.prediction_network(next_hidden)
        return next_hidden, reward, value, policy_logits
    
    def action_to_onehot(self, action):
        """Helper to convert action index to one-hot encoding."""
        if isinstance(action, torch.Tensor):
            if action.dim() == 0:
                action = action.unsqueeze(0)
            action_onehot = torch.zeros(action.size(0), self.action_dim, device=action.device)
            action_onehot.scatter_(1, action.unsqueeze(1).long(), 1)
        else:
            action_onehot = torch.zeros(1, self.action_dim)
            action_onehot[0, action] = 1
        return action_onehot

    def recurrent_inference_batch(self, hidden_batch, action_batch):
        """
        Perform recurrent inference on a batch of hidden states and actions.
        This is more efficient than calling recurrent_inference multiple times.
        
        Args:
            hidden_batch: Batch of hidden states [batch_size, hidden_dim]
            action_batch: Batch of action indices [batch_size]
            
        Returns:
            next_hidden_batch: Batch of next hidden states
            reward_batch: Batch of rewards
            value_batch: Batch of values
            policy_logits_batch: Batch of policy logits
        """
        batch_size = hidden_batch.size(0)
        
        # Convert actions to one-hot encoding
        action_onehot_batch = torch.zeros(batch_size, self.action_dim, device=hidden_batch.device)
        for i in range(batch_size):
            action_onehot_batch[i, action_batch[i]] = 1.0
        
        # Forward pass through dynamics network
        next_hidden_batch, reward_batch = self.dynamics_network(hidden_batch, action_onehot_batch)
        
        # Forward pass through prediction network
        value_batch, policy_logits_batch = self.prediction_network(next_hidden_batch)
        
        return next_hidden_batch, reward_batch, value_batch, policy_logits_batch 