"""
JAX-compatible Gymnasium environment for Narde.
This version uses jax.numpy for array operations and the NardeJAX game implementation.
"""

import gymnasium as gym
import jax
import jax.numpy as jnp
from gymnasium import spaces
import random
from typing import Tuple, Dict, Any, List, Optional, Union
import numpy as np

from gym_narde.envs.narde_jax import NardeJAX, rotate_board

# Make JAX operations deterministic for reproducibility
jax.config.update('jax_platform_name', 'cpu')  # Use CPU for deterministic behavior
jax.config.update('jax_enable_x64', True)     # Use 64-bit precision


class NardeEnvJAX(gym.Env):
    """
    Single-agent Gymnasium environment for Long Nardy in 'white-only' perspective.
    After the agent (White) moves, we rotate the board so the opponent also
    appears as White next turn. The environment continues until one side wins.
    
    This version uses JAX for potentially accelerated operations.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, max_steps=1000, debug=False):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.debug = debug

        # Create the Narde logic with JAX
        self.game = NardeJAX()

        # Define the action space:
        #   We encode a move by (move_index, move_type),
        #   where move_index = from_pos*24 + to_pos,
        #   and move_type=1 => bearing off (to_pos='off').
        self.action_space = spaces.Tuple((
            spaces.Discrete(576),  # 24*24 = 576
            spaces.Discrete(2)     # 0=regular, 1=bear off
        ))

        # Observation space: 24 board points + how many dice remain + borne-off counts + is_doubles
        # We'll store them as a Box for simplicity
        # We fill: obs[:24] = board, obs[24] = len(dice), obs[25]= borne_off_white, obs[26]= borne_off_black, obs[27]= is_doubles?
        # (You can store dice faces more explicitly if you prefer.)
        low = jnp.array([-15]*24 + [0, 0, 0, 0], dtype=jnp.bfloat16)
        high = jnp.array([+15]*24 + [6, 15, 15, 1], dtype=jnp.bfloat16)
        # Convert to numpy for Gymnasium compatibility
        self.observation_space = spaces.Box(
            low=np.array(low), 
            high=np.array(high), 
            dtype=np.bfloat16
        )

        self.dice = []
        self.is_doubles = False
        self.steps_taken = 0
        self.reset()

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to the starting position.
        Returns observation and info dict.
        """
        super().reset(seed=seed)
        
        # Reset game state
        self.game = NardeJAX()
        self.dice = []
        self.is_doubles = False
        self.steps_taken = 0
        
        # Roll initial dice
        self._roll_dice()
        
        # If first player has no valid moves, rotate board and roll again
        if not self.game.get_valid_moves():
            self.game.rotate_board_for_next_player()
            self._roll_dice()
        
        # Get observation and info
        obs = self._get_obs()
        info = {"dice": self.dice.copy()}
        
        return obs, info

    def step(self, action):
        """
        Take one step in the environment, performing the given action.
        """
        move = self._decode_action(action)
        
        if self.debug:
            print(f"DEBUG step: action={action}, decoded to move={move}")
            print(f"DEBUG step: current dice={self.dice}, game dice={self.game.dice}")
        
        # Check if move is valid by comparing with all valid moves
        valid_moves = self.game.get_valid_moves()
        
        # If there are no valid moves available, skip the turn
        if not valid_moves:
            # Rotate the board for the next player
            self.game.rotate_board_for_next_player()
            self._roll_dice()
            return self._get_obs(), 0.0, False, False, {"dice": self.dice.copy(), "skipped_turn": True}
            
        if move not in valid_moves:
            if self.debug:
                print(f"DEBUG step: Invalid move {move}, valid moves are {valid_moves}")
            # If move is invalid, episode is over with big penalty
            return self._get_obs(), -1.0, True, False, {"dice": self.dice.copy(), "invalid_move": True}
        
        # Execute the valid move
        is_bear_off = move[1] == 'off'
        self.game.execute_move(move, self.dice)
        self.steps_taken += 1
        
        # Check for termination
        terminated = self.game.is_game_over() or self.steps_taken >= self.max_steps
        
        # If no dice left, roll for next player
        if not self.dice:
            self.game.rotate_board_for_next_player()
            self._roll_dice()
        
        # Check if game is over (someone won)
        winner = self.game.get_winner()
        
        # Give reward based on game state
        reward = 0.0
        if winner is not None:
            # Big reward if White (us) won
            reward = 1.0 if winner > 0 else -1.0
            terminated = True
        elif is_bear_off:
            # Small reward for bearing off
            reward = 0.1
        
        # Check if next player has valid moves after this
        valid_moves_next = self.game.get_valid_moves()
        
        # If no valid moves and no dice, roll new dice for next player
        if not valid_moves_next and not self.dice:
            # If next player has no valid moves and no dice left,
            # roll new dice to avoid premature termination
            self._roll_dice()
            
            # If still no valid moves, keep skipping until we find a valid move
            # or determine the game should truly end
            max_skip_attempts = 5
            skip_count = 0
            
            while not valid_moves_next and skip_count < max_skip_attempts:
                # Rotate board and roll new dice
                self.game.rotate_board_for_next_player()
                self._roll_dice()
                valid_moves_next = self.game.get_valid_moves()
                skip_count += 1
            
            # If after multiple attempts there are still no valid moves,
            # the game should continue with the next evaluation
            if not valid_moves_next:
                return self._get_obs(), 0.0, False, False, {"dice": self.dice.copy(), "skipped_multiple_turns": True}

        return self._get_obs(), reward, terminated, False, {"dice": self.dice.copy()}

    def _decode_action(self, action):
        """
        Convert (move_index, move_type) to (from_pos, to_pos) or (from_pos, 'off').
        move_index = from_pos*24 + to_pos.
        If move_type==1 => to_pos='off' (bearing off).
        """
        if isinstance(action, tuple):
            move_index, move_type = action
        else:
            # If user only gives a single integer, treat it as (move_index,0)
            move_index = action
            move_type = 0

        from_pos = move_index // 24
        to_pos = move_index % 24

        if move_type == 1:
            # Bearing off => ignore to_pos, use 'off'
            return (from_pos, 'off')
        return (from_pos, to_pos)

    def _roll_dice(self):
        """Roll dice and update dice state."""
        # Roll two dice
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        
        # Check if doubles
        self.is_doubles = (die1 == die2)
        
        # Set dice values
        if self.is_doubles:
            # For doubles, get four moves
            self.dice = [die1, die1, die1, die1]
        else:
            self.dice = [die1, die2]
        
        # Update game dice
        self.game.dice = self.dice.copy()
        
        return self.dice

    def _get_obs(self):
        """
        Return the observation as a single vector:
        - board positions (24 values)
        - number of dice remaining (1 value)
        - borne off counts for white and black (2 values)
        - is_doubles flag (1 value)
        """
        # Create observation using jax.numpy
        obs = jnp.zeros(28, dtype=jnp.bfloat16)
        
        # JAX-compatible way to update observation
        # Fill board state (first 24 values)
        for i in range(24):
            obs = obs.at[i].set(float(self.game.board[i]))
        
        # Fill other values
        obs = obs.at[24].set(float(len(self.dice)))
        obs = obs.at[25].set(float(self.game.borne_off_white))
        obs = obs.at[26].set(float(self.game.borne_off_black))
        obs = obs.at[27].set(float(self.is_doubles))
        
        # Convert to numpy array for Gymnasium compatibility
        return np.array(obs)

    def render(self):
        """Render the environment."""
        if self.render_mode != "human":
            return

        # Simple ASCII rendering
        dice_str = f"Dice: {self.dice}"
        print("=" * 40)
        print(dice_str)
        print("=" * 40)
        
        # Format board
        board = [" " for _ in range(24)]
        for i in range(24):
            value = int(self.game.board[i])
            if value > 0:
                board[i] = f"W{value}"
            elif value < 0:
                board[i] = f"B{abs(value)}"
            else:
                board[i] = "  "
        
        # Print board
        print("13 14 15 16 17 18    19 20 21 22 23 24")
        print(" ".join(f"{board[12 + i]:3s}" for i in range(6)) + "   " + 
              " ".join(f"{board[18 + i]:3s}" for i in range(6)))
        print("-" * 40)
        print(" ".join(f"{board[11 - i]:3s}" for i in range(6)) + "   " + 
              " ".join(f"{board[5 - i]:3s}" for i in range(6)))
        print("12 11 10  9  8  7     6  5  4  3  2  1")
        
        # Print borne off
        print("=" * 40)
        print(f"Borne off - White: {self.game.borne_off_white}, Black: {self.game.borne_off_black}")
        print("=" * 40)

    def close(self):
        """Clean up resources."""
        pass 