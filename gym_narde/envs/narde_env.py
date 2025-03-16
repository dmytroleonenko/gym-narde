import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

from gym_narde.envs.narde import Narde, rotate_board

class NardeEnv(gym.Env):
    """Narde environment implementing gym.Env."""
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None, max_steps=500):
        """Initialize the Narde environment.
        
        Args:
            render_mode (str, optional): The render mode to use. Defaults to None.
            max_steps (int, optional): Maximum number of steps before episode ends. Defaults to 500.
        """
        super().__init__()
        
        # Initialize game
        self.game = Narde()
        self.current_player = 1  # White starts
        self.consecutive_skip_turns = 0
        self.step_count = 0
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Action space: (move_index, move_type)
        # move_index: 0-575 (24*24 possible from-to combinations)
        # move_type: 0 for regular move, 1 for bearing off
        self.action_space = spaces.Tuple((
            spaces.Discrete(576),  # 24*24 possible moves
            spaces.Discrete(2)     # 0: regular move, 1: bearing off
        ))
        
        # Observation space: board (24) + dice (2) + borne_off (2) = 28 features
        self.observation_space = spaces.Box(
            low=np.array([-15] * 24 + [0] * 2 + [0] * 2, dtype=np.float32),
            high=np.array([15] * 24 + [6] * 2 + [15] * 2, dtype=np.float32)
        )
        
        # Initialize dice
        self.dice = [random.randint(1, 6), random.randint(1, 6)]
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state.
        
        Args:
            seed (int, optional): Random seed. Defaults to None.
            options (dict, optional): Additional options. Defaults to None.
        
        Returns:
            numpy.ndarray: Initial observation
        """
        super().reset(seed=seed)
        
        # Reset game state
        self.game = Narde()
        self.current_player = 1
        self.consecutive_skip_turns = 0
        self.step_count = 0
        self.dice = [random.randint(1, 6), random.randint(1, 6)]
        
        return self._get_obs()
    
    def _get_obs(self):
        """Get the current observation."""
        board = self.game.get_perspective_board(self.current_player)
        dice = np.array(self.dice, dtype=np.float32)
        borne_off = np.array([
            self.game.borne_off_white if self.current_player == 1 else self.game.borne_off_black,
            self.game.borne_off_black if self.current_player == 1 else self.game.borne_off_white
        ], dtype=np.float32)
        return np.concatenate([board, dice, borne_off])
    
    def step(self, action):
        """Execute one step in the environment.
        
        Args:
            action: Tuple (move_index, move_type) or integer action code
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Convert action to move
        move = self.action_to_move(action)
        
        # Get valid moves
        valid_moves = self.game.get_valid_moves(self.dice, self.current_player)
        print("Valid moves:", valid_moves)
        
        # Check if move is valid
        if move not in valid_moves:
            print(f"First move {move} not in valid moves: {valid_moves}")
            self.consecutive_skip_turns += 1
            reward = -1.0
        else:
            # Execute move
            print(f"Executing first move: {move}")
            self.game.execute_rotated_move(move, self.current_player)
            self.consecutive_skip_turns = 0
            reward = 0.0
        
        # Increment step counter
        self.step_count += 1
        
        # Check if game is over
        done = (self.game.borne_off_white >= 15 or 
                self.game.borne_off_black >= 15 or 
                self.consecutive_skip_turns >= 4 or
                self.step_count >= self.max_steps)
        
        # Update reward for game over
        if done:
            if self.game.borne_off_white >= 15 and self.current_player == 1:
                reward = 1.0
            elif self.game.borne_off_black >= 15 and self.current_player == -1:
                reward = 1.0
            elif self.consecutive_skip_turns >= 4:
                reward = -1.0
            elif self.step_count >= self.max_steps:
                reward = -1.0
        
        # Switch player and roll dice for next turn
        if not done:
            self.current_player *= -1
            self.dice = [random.randint(1, 6), random.randint(1, 6)]
        
        return self._get_obs(), reward, done, {}
    
    def action_to_move(self, action_code):
        """Convert an action code to a move tuple.
        
        Args:
            action_code: Either an integer representing the action or a tuple (move_index, move_type)
                       where move_type is 0 for regular move, 1 for bearing off
        
        Returns:
            tuple: A tuple of (from_pos, to_pos) representing the move
        """
        # Handle both integer and tuple action codes
        if isinstance(action_code, tuple):
            move_index, move_type = action_code
        else:
            move_index = action_code
            move_type = 0
        
        # Convert action code to positions
        from_pos = move_index // 24
        to_pos = move_index % 24
        
        # For bearing off moves
        if move_type == 1:
            to_pos = 'off'
        
        # For Black's perspective, we need to rotate positions
        if self.current_player == -1:
            # Convert from Black's perspective to White's perspective
            if to_pos == 'off':
                from_pos = (from_pos + 12) % 24
            else:
                from_pos = (from_pos + 12) % 24
                to_pos = (to_pos + 12) % 24
        
        return (from_pos, to_pos)

    def render(self):
        if self.render_mode == "human":
            board_str = ""
            for i in range(24):
                board_str += f"{self.game.board[i]:>3} "
                if (i + 1) % 6 == 0:
                    board_str += "\n"
            print(board_str)

    def close(self):
        pass
        
    def _check_game_ended(self):
        # Check if any player has borne off all checkers
        if self.game.borne_off_white >= 15 or self.game.borne_off_black >= 15:
            # Calculate reward based on mars/oin rules
            white_off = self.game.borne_off_white
            black_off = self.game.borne_off_black

            if white_off >= 15:
                reward = 2 if black_off == 0 else 1  # Mars if opponent has none, else Oin
            else:
                reward = 2 if white_off == 0 else 1

            return True, reward
        return False, 0

    def rotate_action(self, action_code):
        """Rotate action code for Black's perspective.
        
        Args:
            action_code (int): An integer representing the action
            
        Returns:
            int: The rotated action code
        """
        # Convert action code to positions
        from_pos = action_code // 24
        to_pos = action_code % 24
        
        # Don't rotate bearing off moves (to_pos == 0)
        if to_pos == 0:
            # Only rotate from_pos
            rotated_from = (from_pos + 12) % 24
            return rotated_from * 24
        
        # Rotate both positions
        rotated_from = (from_pos + 12) % 24
        rotated_to = (to_pos + 12) % 24
        return rotated_from * 24 + rotated_to
