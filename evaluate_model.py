import os
import gymnasium as gym  # Updated to gymnasium instead of gym
import torch
import numpy as np
import gym_narde  # Important: Import the custom environment
from train_deepq_pytorch import DecomposedDQN, create_moves_from_action
from gym_narde.envs.narde import rotate_board  # Import for board rotation
import random
import time
import argparse

# Define player constants
WHITE = 1
BLACK = -1
COLORS = {WHITE: "White", BLACK: "Black"}

class RandomAgent:
    def __init__(self, color):
        self.color = color
        self.name = f'Random({COLORS[self.color]})'
    
    def choose_best_action(self, env, dice):
        valid_moves = env.game.get_valid_moves(dice, self.color)
        
        if len(valid_moves) == 0:
            return (0, 0)
        
        # Choose random valid moves
        move1 = random.choice(valid_moves)
        move2 = random.choice(valid_moves) if len(valid_moves) > 1 else move1
        
        # Convert moves to action format
        from_pos1, to_pos1 = move1
        from_pos2, to_pos2 = move2
        
        move1_code = from_pos1 * 24 + (0 if to_pos1 == 'off' else to_pos1)
        move2_code = from_pos2 * 24 + (0 if to_pos2 == 'off' else to_pos2)
        
        return (move1_code, move2_code)

class AIAgent:
    def __init__(self, model_path, color):
        self.color = color
        self.name = f'AI({COLORS[self.color]})'
        
        # Determine device (CUDA, MPS or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"AI using device: {self.device}")
        
        # Load the model
        self.state_size = 24
        self.move_space_size = 576  # 24x24 possible moves
        self.model = DecomposedDQN(self.state_size, self.move_space_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def choose_best_action(self, env, dice):
        state = env.unwrapped.game.get_perspective_board(self.color)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get valid moves
        valid_moves = env.unwrapped.game.get_valid_moves(dice, self.color)
        
        if len(valid_moves) == 0:
            return (0, 0)
            
        # Convert valid moves to action codes for comparison
        valid_move_combos = []
        valid_first_moves = {}
        
        for move1 in valid_moves:
            from_pos1, to_pos1 = move1
            
            # Convert move to code format
            if to_pos1 == 'off':
                # Special case for bearing off
                move1_code = from_pos1 * 24  # Use position 0 as placeholder for 'off'
            else:
                move1_code = from_pos1 * 24 + to_pos1
                
            # Track valid second moves for each first move
            valid_first_moves[move1_code] = []
            
            # No second move case
            valid_move_combos.append((move1_code, 0))
            valid_first_moves[move1_code].append(0)
        
        # Get Q-values from model for first move
        with torch.no_grad():
            move1_q_values = self.model(state_tensor)
            
            # Extract Q-values for valid first moves
            valid_move1_indices = list(valid_first_moves.keys())
            
            # Convert to tensor for indexing
            valid_move1_tensor = torch.tensor(valid_move1_indices, device=self.device)
            valid_move1_q_values = move1_q_values.squeeze(0).index_select(0, valid_move1_tensor)
            
            # Find best first move
            best_move1_idx = torch.argmax(valid_move1_q_values).item()
            best_move1_code = valid_move1_indices[best_move1_idx]
            
            # Get valid second moves for this first move
            valid_move2_codes = valid_first_moves[best_move1_code]
            
            # If no valid second moves, return just the first move
            if not valid_move2_codes or len(valid_move2_codes) == 0:
                return (best_move1_code, 0)
                
            # Get Q-values for second moves based on selected first move
            selected_move1 = torch.tensor([best_move1_code], device=self.device)
            move2_q_values = self.model(state_tensor, selected_move1)
            
            # Extract Q-values for valid second moves
            valid_move2_tensor = torch.tensor(valid_move2_codes, device=self.device)
            valid_move2_q_values = move2_q_values.squeeze(0).index_select(0, valid_move2_tensor)
            
            # Find best second move
            best_move2_idx = torch.argmax(valid_move2_q_values).item()
            best_move2_code = valid_move2_codes[best_move2_idx]
            
            return (best_move1_code, best_move2_code)
            
        # Fallback to random valid move if something goes wrong
        move1 = random.choice(valid_moves)
        from_pos1, to_pos1 = move1
        move1_code = from_pos1 * 24 + (0 if to_pos1 == 'off' else to_pos1)
        
        return (move1_code, 0)

def evaluate(model_name='narde_model_final.pt', num_games=100, render=False):
    # Create environment
    env = gym.make('gym_narde:narde-v0', render_mode='human' if render else None)
    
    # Set up AI agent
    model_path = os.path.join(os.getcwd(), 'saved_models', model_name)
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    # Statistics
    wins = {WHITE: 0, BLACK: 0}
    game_lengths = []
    
    # Play games
    for game in range(num_games):
        # Randomly assign colors
        ai_color = WHITE if np.random.rand() > 0.5 else BLACK
        random_color = BLACK if ai_color == WHITE else WHITE
        
        agents = {
            ai_color: AIAgent(model_path, ai_color),
            random_color: RandomAgent(random_color)
        }
        
        if render and game == 0:
            print(f"Game {game+1}: AI is {COLORS[ai_color]}, Random is {COLORS[random_color]}")
        
        # Initialize the game
        observation, _ = env.reset()
        current_player = env.unwrapped.current_player
        steps = 0
        
        # Game loop
        while True:
            steps += 1
            player = agents[current_player]
            
            # Roll dice
            dice = [np.random.randint(1, 7), np.random.randint(1, 7)]
            
            if render and game == 0:
                print(f"\nTurn {steps}: {player.name} rolls {dice}")
            
            # Get action from current player
            action = player.choose_best_action(env.unwrapped, dice)
            
            # Take step in environment
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if done:
                winner = env.unwrapped.current_player
                wins[winner] += 1
                game_lengths.append(steps)
                
                if render:
                    if ai_color == winner:
                        print(f"Game {game+1}: AI ({COLORS[ai_color]}) wins in {steps} steps")
                    else:
                        print(f"Game {game+1}: Random ({COLORS[random_color]}) wins in {steps} steps")
                break
            
            current_player = env.unwrapped.current_player
        
        # Print progress every 10 games
        if (game + 1) % 10 == 0:
            ai_wins = wins[ai_color] if game == 0 else sum([1 for i in range(game+1) if (i % 2 == 0 and wins[WHITE] > 0) or (i % 2 == 1 and wins[BLACK] > 0)])
            win_rate = ai_wins / (game + 1) * 100
            print(f"Progress: {game+1}/{num_games} games, AI win rate: {win_rate:.1f}%")
    
    # Calculate final statistics
    ai_win_count = sum([wins[WHITE if game % 2 == 0 else BLACK] for game in range(num_games)])
    ai_win_rate = ai_win_count / num_games * 100
    avg_game_length = sum(game_lengths) / len(game_lengths)
    
    print("\n=== Evaluation Results ===")
    print(f"Model: {model_name}")
    print(f"Games played: {num_games}")
    print(f"AI win rate: {ai_win_rate:.1f}%")
    print(f"Average game length: {avg_game_length:.1f} steps")
    print("=" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN agent against a random agent')
    parser.add_argument('--model', type=str, default='narde_model_final.pt', 
                        help='Name of the model file in the saved_models directory')
    parser.add_argument('--games', type=int, default=100, 
                        help='Number of games to play')
    parser.add_argument('--render', action='store_true',
                        help='Render the first game')
    
    args = parser.parse_args()
    
    evaluate(model_name=args.model, num_games=args.games, render=args.render)