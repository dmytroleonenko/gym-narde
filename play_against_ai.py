import os
import gymnasium as gym
import torch
import numpy as np
import time
import gym_narde  # Important: Import the custom environment
from train_deepq_pytorch import DecomposedDQN, create_moves_from_action, check_move_valid

# Define player constants
WHITE = 1
BLACK = -1
COLORS = {WHITE: "White", BLACK: "Black"}
TOKEN = {WHITE: "○", BLACK: "●"}

class AIPlayer:
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
        move1 = np.random.choice(valid_moves)
        from_pos1, to_pos1 = move1
        move1_code = from_pos1 * 24 + (0 if to_pos1 == 'off' else to_pos1)
        
        return (move1_code, 0)

class HumanPlayer:
    def __init__(self, color):
        self.color = color
        self.name = f'Human({COLORS[self.color]})'
    
    def choose_best_action(self, env, dice):
        # Display the board for the human player
        print(f"\nYour turn ({TOKEN[self.color]} - {COLORS[self.color]})")
        if hasattr(env, 'render'):
            env.render()
        print(f"Dice rolls: {dice}")
        
        # Get valid moves
        valid_moves = env.unwrapped.game.get_valid_moves(dice, self.color)
        
        if len(valid_moves) == 0:
            print("No valid moves. Skipping turn.")
            return (0, 0)
        
        # Display valid moves
        print("\nValid moves:")
        for i, move in enumerate(valid_moves):
            from_pos, to_pos = move
            to_str = 'off' if to_pos == 'off' else str(to_pos + 1)
            print(f"{i+1}. From point {from_pos + 1} to point {to_str}")
        
        # Get first move from user
        while True:
            try:
                choice = input("\nEnter the number of your first move (or 0 to skip): ")
                if not choice:  # Handle empty input gracefully
                    print("Empty input, selecting first valid move")
                    move1 = valid_moves[0]
                    break
                
                move1_idx = int(choice) - 1
                if move1_idx == -1:
                    return (0, 0)
                
                if 0 <= move1_idx < len(valid_moves):
                    move1 = valid_moves[move1_idx]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number.")
                # In case of EOF error or other issues, default to first valid move
                move1 = valid_moves[0]
                break
            except EOFError:
                print("Input error, selecting first valid move")
                move1 = valid_moves[0]
                break
        
        # After the first move, get remaining valid moves
        # In a real implementation, we would need to simulate the first move 
        # and then calculate the valid moves for the second move
        print("\nFor demonstration, selecting a random second move...")
        move2 = valid_moves[0]  # For simplicity
        
        # Convert moves to action format
        from_pos1, to_pos1 = move1
        from_pos2, to_pos2 = move2
        
        move1_code = from_pos1 * 24 + (0 if to_pos1 == 'off' else to_pos1)
        move2_code = from_pos2 * 24 + (0 if to_pos2 == 'off' else to_pos2)
        
        return (move1_code, move2_code)

def play_game(model_name='narde_model_final.pt'):
    # Create environment
    env = gym.make('gym_narde:narde-v0', render_mode='human')
    
    # Set up players
    model_path = os.path.join(os.getcwd(), 'saved_models', model_name)
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    # Randomly assign colors
    human_color = WHITE if np.random.rand() > 0.5 else BLACK
    ai_color = BLACK if human_color == WHITE else WHITE
    
    players = {
        human_color: HumanPlayer(human_color),
        ai_color: AIPlayer(model_path, ai_color)
    }
    
    print(f"\nYou are playing as {COLORS[human_color]} ({TOKEN[human_color]})")
    print(f"AI is playing as {COLORS[ai_color]} ({TOKEN[ai_color]})")
    
    # Initialize the game
    observation, _ = env.reset()
    current_player = env.unwrapped.current_player
    
    wins = {WHITE: 0, BLACK: 0}
    t = time.time()
    
    # Main game loop
    for i in range(1000):  # Max steps
        player = players[current_player]
        print(f"\nCurrent player: {player.name}")
        
        # Roll dice
        dice = [np.random.randint(1, 7), np.random.randint(1, 7)]
        print(f"Dice rolled: {dice}")
        
        # Get action from current player
        action = player.choose_best_action(env, dice)
        
        # Take step in environment
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Display the board
        if hasattr(env, 'render'):
            env.render()
        
        if done:
            winner = env.unwrapped.current_player
            wins[winner] += 1
            
            print(f"\nGame over! Winner: {players[winner].name}")
            print(f"Duration: {time.time() - t:.3f} seconds")
            break
        
        current_player = env.unwrapped.current_player

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Narde against a trained AI')
    parser.add_argument('--model', type=str, default='narde_model_final.pt', 
                        help='Name of the model file in the saved_models directory')
    
    args = parser.parse_args()
    play_game(model_name=args.model)