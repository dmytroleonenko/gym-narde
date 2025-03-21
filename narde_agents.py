#!/usr/bin/env python3
"""
Standardized interface for agents that can play the Narde game.
This module contains:
- A base NardeAgent class that defines the interface for all agents
- RandomAgent implementation that makes random moves
- DQNAgent implementation that uses a trained DQN model
"""

import numpy as np
import torch
import random
from abc import ABC, abstractmethod
from train_simpledqn import DQN  # Import DQN model class from training script
from benchmarks.narde_benchmark import GenericNardeAgent
import torch.nn as nn
import torch.optim as optim

class NardeAgent(ABC):
    """
    Abstract base class for Narde agents.
    
    All agents must implement the select_action method that takes the current
    environment state and returns an action.
    """
    
    def __init__(self, name="Agent"):
        """Initialize the agent with a name for identification."""
        self.name = name
    
    @abstractmethod
    def select_action(self, env, state, is_white):
        """
        Select an action based on the current state.
        
        Args:
            env: The Narde environment (unwrapped)
            state: The current state observation (tensor or array)
            is_white: Boolean indicating if the agent is playing as White
            
        Returns:
            action: The selected action in the format expected by the env.step() method
        """
        pass
    
    def reset(self):
        """Reset the agent's state between episodes if needed."""
        pass


class RandomAgent(GenericNardeAgent):
    """
    Agent that selects random valid actions.
    """
    
    def __init__(self, name="RandomAgent"):
        """
        Initialize the random agent.
        
        Args:
            name: Name identifier for the agent
        """
        self._name = name
    
    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return self._name
    
    def reset(self):
        """Reset the agent's state for a new game."""
        # Nothing to reset for RandomAgent
        pass
    
    def select_action(self, env, state):
        """
        Select a random valid action.
        
        Args:
            env: The Narde environment
            state: Current state observation
            
        Returns:
            The selected action
        """
        valid_moves = env.game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
            
        # Select a random move
        move = random.choice(valid_moves)
        
        # Convert move to action format
        if move[1] == 'off':
            # Bear off move: (from_pos, 'off')
            action = (move[0] * 24, 1)  # action_type=1 for bear-off
        else:
            # Regular move: (from_pos, to_pos)
            action = (move[0] * 24 + move[1], 0)  # action_type=0 for regular move
            
        return action


class DQNAgent(GenericNardeAgent):
    """
    Deep Q-Network agent for playing Narde.
    """
    
    def __init__(self, model_path, epsilon=0.0, name="DQNAgent"):
        """
        Initialize the DQN agent.
        
        Args:
            model_path: Path to the trained model weights
            epsilon: Exploration rate (0=no exploration, 1=random)
            name: Name identifier for the agent
        """
        self._name = name
        self.epsilon = epsilon
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else
                                 "cpu")
        
        print(f"Using device: {self.device}")
        
        # Load the model
        try:
            # First load the state dict to inspect dimensions
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Try to determine the input dimension from the first layer weight
            input_dim = None
            if "network.0.weight" in state_dict:
                input_dim = state_dict["network.0.weight"].shape[1]
                print(f"Detected input dimension: {input_dim}")
            elif "0.weight" in state_dict:
                input_dim = state_dict["0.weight"].shape[1]
                print(f"Detected input dimension: {input_dim}")
                
            if input_dim is None:
                # Fallback to default
                input_dim = 28  # Based on the error message
                print(f"Couldn't detect input dimension, using default: {input_dim}")
                
            # Set the output dimension based on the model or use default
            output_dim = 1152  # 24*24*2 (from_pos, to_pos, action_type)
            
            # Create the network with correct dimensions
            self.q_network = DQN(input_dim, output_dim).to(self.device)
            
            # Load the weights
            if all(key.startswith("network.") for key in state_dict.keys()):
                # Direct state dict for the network
                self.q_network.load_state_dict(state_dict)
            else:
                # Try loading into the network component
                self.q_network.network.load_state_dict(state_dict)
                
            self.q_network.eval()  # Set to evaluation mode
            print(f"Successfully loaded model from {model_path}")
            
            # Store the input dimension for later use
            self.state_dim = input_dim
            self.action_dim = output_dim
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise
    
    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return self._name
    
    def reset(self):
        """Reset the agent's state for a new game."""
        # Nothing to reset for DQNAgent
        pass
    
    def _preprocess_state(self, state):
        """
        Preprocess the state for input to the Q-network.
        
        Args:
            state: Raw state from the environment
            
        Returns:
            Preprocessed state tensor
        """
        # Check if we need to adjust state dimension
        if len(state) != self.state_dim:
            print(f"Warning: State dimension mismatch. State: {len(state)}, Expected: {self.state_dim}")
            # Simple solution: truncate or pad as needed
            if len(state) > self.state_dim:
                # Truncate to match expected input
                state = state[:self.state_dim]
            else:
                # Pad with zeros
                padding = np.zeros(self.state_dim - len(state))
                state = np.concatenate([state, padding])
        
        # Convert to torch tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return state_tensor
    
    def _get_action_mask(self, env):
        """
        Create a mask for valid actions.
        
        Args:
            env: The Narde environment
            
        Returns:
            Binary mask where 1=valid action, 0=invalid action
        """
        valid_moves = env.game.get_valid_moves()
        
        # Initialize all actions as invalid
        action_mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
        
        # Mark valid actions as 1
        for move in valid_moves:
            if move[1] == 'off':
                # Bear off move
                action_idx = move[0] * 24
                action_mask[action_idx] = True
            else:
                # Regular move
                action_idx = move[0] * 24 + move[1]
                action_mask[action_idx] = True
                
        return action_mask
    
    def select_action(self, env, state):
        """
        Select an action using the Q-network.
        
        Args:
            env: The Narde environment
            state: Current state observation
            
        Returns:
            The selected action
        """
        # With probability epsilon, select random action
        if random.random() < self.epsilon:
            valid_moves = env.game.get_valid_moves()
            if not valid_moves:
                raise ValueError("No valid moves available")
                
            # Select a random move
            move = random.choice(valid_moves)
            
            # Convert move to action format
            if move[1] == 'off':
                # Bear off move: (from_pos, 'off')
                action = (move[0] * 24, 1)  # action_type=1 for bear-off
            else:
                # Regular move: (from_pos, to_pos)
                action = (move[0] * 24 + move[1], 0)  # action_type=0 for regular move
                
            return action
        
        # Otherwise, use the Q-network
        state_tensor = self._preprocess_state(state)
        
        # Get valid action mask
        action_mask = self._get_action_mask(env)
        
        # Get Q-values for all actions
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            masked_q_values = q_values.masked_fill(~action_mask, float('-inf'))
            masked_q_values = masked_q_values.squeeze(0)  # Remove batch dimension
            
            # Select action with highest Q-value
            best_action_idx = torch.argmax(masked_q_values).item()
            
        # Convert to action format (from_pos, action_type)
        from_pos = best_action_idx // 24
        to_pos = best_action_idx % 24
        
        if best_action_idx in range(from_pos * 24, from_pos * 24 + 24) and not action_mask[best_action_idx]:
            # If we somehow selected an invalid move despite masking
            # This is a safety check that shouldn't normally be triggered
            valid_moves = env.game.get_valid_moves()
            
            if not valid_moves:
                raise ValueError("No valid moves available")
            
            move = random.choice(valid_moves)
            if move[1] == 'off':
                return (move[0] * 24, 1)
            else:
                return (move[0] * 24 + move[1], 0)
            
        # Check if this is a bear off move
        # We need to infer this from the valid moves
        valid_moves = env.game.get_valid_moves()
        for move in valid_moves:
            if move[0] == from_pos and move[1] == 'off':
                return (from_pos * 24, 1)  # Bear off move
            elif move[0] == from_pos and move[1] == to_pos:
                return (from_pos * 24 + to_pos, 0)  # Regular move
        
        # If we reach here, something went wrong
        raise ValueError(f"Invalid action selected: {best_action_idx}") 