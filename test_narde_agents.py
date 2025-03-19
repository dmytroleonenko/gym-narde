#!/usr/bin/env python3
"""
Tests for NardeAgent classes.
"""

import pytest
import numpy as np
import torch
import random
import os
from unittest.mock import MagicMock, patch

# Import agents
from narde_agents import NardeAgent, RandomAgent, DQNAgent


class TestRandomAgent:
    """Tests for the RandomAgent class."""
    
    def test_init(self):
        """Test agent initialization."""
        agent = RandomAgent()
        assert agent.name == "RandomAgent"
        
        agent = RandomAgent(name="TestAgent")
        assert agent.name == "TestAgent"
    
    def test_reset(self):
        """Test reset method."""
        agent = RandomAgent()
        # Reset should not raise errors
        agent.reset()
    
    def test_select_action_with_valid_moves(self):
        """Test that agent can select a random action when valid moves exist."""
        # Create a mock environment
        mock_env = MagicMock()
        mock_env.game.get_valid_moves.return_value = [(0, 1), (5, 6)]
        
        # Create agent and select action
        agent = RandomAgent()
        action = agent.select_action(mock_env, None, True)
        
        # Check that action is in the expected format
        assert isinstance(action, tuple)
        assert len(action) == 2
        assert isinstance(action[0], int)
        assert isinstance(action[1], int)
        
        # Verify the mock was called
        mock_env.game.get_valid_moves.assert_called_once()
    
    def test_select_action_with_bear_off(self):
        """Test that agent handles bear off moves properly."""
        # Create a mock environment
        mock_env = MagicMock()
        mock_env.game.get_valid_moves.return_value = [(15, 'off')]
        
        # Create agent and select action
        agent = RandomAgent()
        action = agent.select_action(mock_env, None, True)
        
        # Check the action format for bear off 
        assert action == (15 * 24 + 0, 1)
        
        # Verify the mock was called
        mock_env.game.get_valid_moves.assert_called_once()
    
    def test_select_action_with_no_valid_moves(self):
        """Test that agent raises ValueError when no valid moves exist."""
        # Create a mock environment
        mock_env = MagicMock()
        mock_env.game.get_valid_moves.return_value = []
        
        # Create agent
        agent = RandomAgent()
        
        # Check that it raises ValueError
        with pytest.raises(ValueError, match="No valid moves available"):
            agent.select_action(mock_env, None, True)
        
        # Verify the mock was called
        mock_env.game.get_valid_moves.assert_called_once()


class TestDQNAgent:
    """Tests for the DQNAgent class."""
    
    @pytest.fixture
    def mock_dqn_model(self):
        """Create a mock DQN model."""
        with patch('narde_agents.DQN', autospec=True) as mock_dqn:
            # Configure the mock
            mock_instance = mock_dqn.return_value
            mock_instance.to.return_value = mock_instance
            mock_instance.eval.return_value = None
            
            # Configure forward method to return a tensor with 1152 elements (24*24*2)
            mock_q_values = torch.zeros(1, 1152)
            mock_q_values[0, 100] = 1.0  # Set a high value at index 100
            mock_instance.return_value = mock_q_values
            
            yield mock_dqn
    
    @pytest.fixture
    def mock_torch_load(self):
        """Mock torch.load to avoid loading a real model file."""
        with patch('torch.load', autospec=True) as mock_load:
            mock_load.return_value = {}  # Return an empty state dict
            yield mock_load
    
    def test_init(self, mock_dqn_model, mock_torch_load):
        """Test agent initialization."""
        agent = DQNAgent("fake_model.pth")
        assert agent.name == "DQNAgent"
        assert agent.model_path == "fake_model.pth"
        assert agent.model is None  # Model should be lazy-loaded
        
        agent = DQNAgent("fake_model.pth", name="TestDQN")
        assert agent.name == "TestDQN"
    
    def test_reset(self, mock_dqn_model, mock_torch_load):
        """Test reset method."""
        agent = DQNAgent("fake_model.pth")
        # Reset should not raise errors
        agent.reset()
    
    def test_load_model(self, mock_dqn_model, mock_torch_load):
        """Test model loading."""
        agent = DQNAgent("fake_model.pth")
        agent._load_model(24, 1152)
        
        # Check that the model was loaded
        assert agent.model is not None
        assert agent.state_dim == 24
        assert agent.action_dim == 1152
        
        # Verify mocks were called
        mock_dqn_model.assert_called_once_with(24, 1152)
        mock_torch_load.assert_called_once()
    
    def test_select_action_as_white(self, mock_dqn_model, mock_torch_load):
        """Test action selection as White player."""
        # Create a mock environment
        mock_env = MagicMock()
        mock_env.game.get_valid_moves.return_value = [(0, 1), (5, 6)]
        mock_env.action_space = [MagicMock(), MagicMock()]
        mock_env.action_space[0].n = 24
        mock_env.action_space[1].n = 48
        
        # Create a fake state
        state = np.zeros(24)
        
        # Create agent
        agent = DQNAgent("fake_model.pth")
        
        # Mock the model methods to avoid actual computation
        agent._load_model = MagicMock()
        agent.model = MagicMock()
        agent.action_dim = 1152
        
        # Configure model to return q-values
        q_values = torch.zeros(1, 1152)
        q_values[0, 100] = 1.0  # Set a high value at a valid action index
        agent.model.return_value = q_values
        
        # Select action
        action = agent.select_action(mock_env, state, True)
        
        # Check action format
        assert isinstance(action, tuple)
        assert len(action) == 2
        assert isinstance(action[0], int)
        assert isinstance(action[1], int)
        
        # Verify the mock was called
        mock_env.game.get_valid_moves.assert_called_once()
        agent.model.assert_called_once()
    
    def test_select_action_as_black(self, mock_dqn_model, mock_torch_load):
        """Test action selection as Black player."""
        # Create a mock environment with a board
        mock_env = MagicMock()
        mock_env.game.get_valid_moves.return_value = [(0, 1)]
        mock_env.action_space = [MagicMock(), MagicMock()]
        mock_env.action_space[0].n = 24
        mock_env.action_space[1].n = 48
        
        # Create a fake state
        state = np.zeros(24)
        
        # Create agent
        agent = DQNAgent("fake_model.pth")
        
        # Mock the model methods to avoid actual computation
        agent._load_model = MagicMock()
        agent.model = MagicMock()
        agent.action_dim = 1152
        
        # Configure model to return q-values
        q_values = torch.zeros(1, 1152)
        q_values[0, 100] = 1.0  # Set a high value at a valid action index
        agent.model.return_value = q_values
        
        # Select action as Black - should use the same path as White now
        action = agent.select_action(mock_env, state, False)
        
        # Check action format
        assert isinstance(action, tuple)
        assert len(action) == 2
        assert isinstance(action[0], int)
        assert isinstance(action[1], int)
        
        # Verify the mocks were called - same verification as for White
        mock_env.game.get_valid_moves.assert_called_once()
        agent.model.assert_called_once()


if __name__ == "__main__":
    pytest.main() 