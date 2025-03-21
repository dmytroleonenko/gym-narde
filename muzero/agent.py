#!/usr/bin/env python3
"""
MuZero agent implementation for Narde.
"""

import torch
import numpy as np
from muzero.models import MuZeroNetwork
from muzero.mcts import MCTS
from muzero.training import get_valid_action_indices


class MuZeroAgent:
    """
    MuZero agent for playing Narde.
    
    Uses a trained MuZero model with MCTS for planning.
    """
    
    def __init__(
        self,
        model_path,
        hidden_dim=256,
        num_simulations=50,
        temperature=0.0,
        device="auto",
        name="MuZeroAgent"
    ):
        """
        Initialize the MuZero agent.
        
        Args:
            model_path: Path to the trained model weights
            hidden_dim: Hidden dimension size used in the model
            num_simulations: Number of MCTS simulations to run
            temperature: Temperature for action selection (0=deterministic)
            device: Device to run the model on
            name: Name identifier for the agent
        """
        self._name = name
        self.temperature = temperature
        self.num_simulations = num_simulations
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"MuZero agent using device: {self.device}")
        
        # Create model with correct dimensions
        self.input_dim = 28  # NardeEnv observation space size
        self.action_dim = 24 * 24  # From-to position pairs
        
        # Load the MuZero network
        self.network = MuZeroNetwork(
            input_dim=self.input_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Load the model weights
        try:
            # Load state dict with weights_only=False to handle PyTorch 2.6+ security changes
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            self.network.load_state_dict(state_dict)
            self.network.eval()
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            # Try alternative loading method for backward compatibility
            try:
                print("Attempting alternative loading method...")
                # Add safe globals for numpy scalar (needed for PyTorch 2.6+)
                import numpy as np
                from torch.serialization import add_safe_globals
                try:
                    add_safe_globals([np._core.multiarray.scalar])
                except AttributeError:
                    # Handle case where numpy structure is different
                    print("Could not access np._core.multiarray.scalar, trying alternative approach")
                
                # Retry loading with weights_only=True
                state_dict = torch.load(model_path, map_location=self.device)
                self.network.load_state_dict(state_dict)
                self.network.eval()
                print(f"Successfully loaded model with alternative method from {model_path}")
            except Exception as e2:
                print(f"Error with alternative loading method: {e2}")
                raise
            
        # Create MCTS for planning
        self.mcts = MCTS(
            network=self.network,
            num_simulations=num_simulations,
            discount=0.997,
            action_space_size=self.action_dim,
            device=self.device
        )
    
    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return self._name
    
    def reset(self):
        """Reset the agent's state for a new game."""
        # Nothing to reset for MuZeroAgent
        pass
    
    def select_action(self, env, state):
        """
        Select an action using MCTS with the MuZero network.
        
        Args:
            env: The Narde environment
            state: Current state observation
            
        Returns:
            The selected action in the format (move_index, move_type)
        """
        # Get valid actions
        valid_actions = get_valid_action_indices(env)
        
        if not valid_actions:
            raise ValueError("No valid moves available")
            
        # Convert numpy array to torch tensor if necessary
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state.to(self.device)
            
        # Run MCTS to get a policy
        mcts_policy = self.mcts.run(
            observation=state_tensor,
            valid_actions=valid_actions,
            add_exploration_noise=False  # No exploration during evaluation
        )
        
        # Select action based on the policy
        if self.temperature == 0:
            # Deterministic selection
            action_idx = np.argmax(mcts_policy)
        else:
            # Stochastic selection with temperature
            policy = mcts_policy ** (1 / self.temperature)
            policy /= policy.sum()
            action_idx = np.random.choice(len(mcts_policy), p=policy)
            
        # Convert action index to action tuple (move_index, move_type)
        # For Narde, we need to determine if it's a bear-off action
        # We'll check the valid moves to see if this is a bear-off
        move_type = 0  # Default to regular move
        from_pos = action_idx // 24
        
        for move in env.game.get_valid_moves():
            if move[0] == from_pos and move[1] == 'off':
                move_type = 1  # It's a bear-off
                break
                
        action = (action_idx, move_type)
        return action 