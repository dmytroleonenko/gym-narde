from flask import Flask, render_template, jsonify, request, session, url_for
import os
import sys
import numpy as np
import torch
import json
import random
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_narde.envs.narde import Narde
from gym_narde.envs.narde_env import NardeEnv
from web.narde_patched import NardePatched
from my_game.narde_game_manager import NardeGameManager

# Import DQN model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from train_deepq_pytorch import DecomposedDQN

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Game sessions storage
games = {}

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          'saved_models', 'narde_model_final.pt')

# Load AI model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ai_model = DecomposedDQN(24, 576)  # State size 24, move space size 24*24=576
ai_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
ai_model.eval()
ai_model.to(device)

# Helper function to log board state - only logs when debug is enabled
def log_board_state(board, message=""):
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"{message} Board state: {board.tolist()}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    logging.info("Server shutdown requested")
    # Get feedback if provided
    data = request.get_json() or {}
    feedback = data.get('feedback', '')
    
    if feedback:
        logging.info(f"User final feedback received: {feedback}")
        # Save feedback to a file, appending if the file already exists
        with open("user_feedback.txt", "a") as f:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] FINAL FEEDBACK (SHUTDOWN): {feedback}\n\n")
    
    import os
    try:
        # Flask 2.x
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            # Flask 1.x
            os._exit(0)
        func()
        return jsonify({'status': 'success', 'message': 'Server shutting down with feedback saved'})
    except Exception as e:
        logging.error(f"Error shutting down: {e}")
        # Force shutdown
        os._exit(0)

@app.route('/api/log', methods=['POST'])
def client_log():
    data = request.get_json()
    if 'message' in data:
        logging.info(data['message'])
    return jsonify({'status': 'ok'})

@app.route('/api/save_feedback', methods=['POST'])
def save_feedback():
    data = request.get_json() or {}
    feedback = data.get('feedback', '')
    
    if feedback:
        logging.info(f"User feedback received: {feedback}")
        
        # Save feedback to a file, appending if the file already exists
        with open("user_feedback.txt", "a") as f:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {feedback}\n\n")
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback saved successfully'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'No feedback provided'
        })

@app.route('/api/new_game', methods=['POST'])
def new_game():
    # Create a new game session
    game_id = str(random.randint(10000, 99999))
    while game_id in games:
        game_id = str(random.randint(10000, 99999))
    
    # Initialize game environment
    env = NardeEnv()
    env.reset()
    
    # Replace the game object with our patched version for better logging
    env.game = NardePatched(env.game)
    
    # Create a game manager
    game_manager = NardeGameManager(env)
    
    # Log how the board is set up initially
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"White checkers at start: Position {np.where(env.game.board > 0)[0].tolist()}")
        logging.debug(f"Black checkers at start: Position {np.where(env.game.board < 0)[0].tolist()}")
        log_board_state(env.game.board, "Initial board state:")
    
    # Store game manager in session
    games[game_id] = game_manager
    session['game_id'] = game_id
    
    # Roll dice for first move
    dice, valid_moves_by_piece = game_manager.roll_dice('white')
    
    return jsonify({
        'game_id': game_id,
        'board': env.game.board.tolist(),
        'current_player': 'white',
        'dice': dice,
        'valid_moves_by_piece': valid_moves_by_piece,
        'borne_off': {
            'white': env.game.borne_off_white,
            'black': env.game.borne_off_black
        }
    })

@app.route('/api/get_valid_moves', methods=['POST'])
def get_valid_moves():
    data = request.get_json()
    game_id = session.get('game_id')
    from_pos = int(data.get('position'))
    
    if not game_id or game_id not in games:
        return jsonify({'error': 'No active game'}), 400
    
    game_manager = games[game_id]
    valid_to_positions = game_manager.get_valid_moves_for_position(from_pos, 'white')
    
    return jsonify({
        'valid_to_positions': valid_to_positions
    })

@app.route('/api/make_move', methods=['POST'])
def make_move():
    data = request.get_json()
    game_id = session.get('game_id')
    from_pos = int(data.get('from_position'))
    to_pos = data.get('to_position')
    
    # Enhanced logging for tracking
    logging.info(f"=== MAKE_MOVE REQUEST ===")
    logging.info(f"From position: {from_pos}, To position: {to_pos}")
    
    # Convert -1 back to 'off' for bearing off
    if to_pos == -1:
        to_pos = 'off'
    else:
        to_pos = int(to_pos)
    
    if not game_id or game_id not in games:
        logging.error(f"No active game with ID {game_id}")
        return jsonify({'error': 'No active game'}), 400
    
    game_manager = games[game_id]
    
    # Execute the move
    result = game_manager.make_move(from_pos, to_pos, 'white')
    
    # Check for errors
    if 'error' in result:
        return jsonify({'error': result['error']}), 400
    
    # If turn is complete, proceed to AI's turn
    if result.get('turn_complete', False):
        return ai_turn(game_id)
    
    # Otherwise, return the result for next move
    return jsonify(result)

def ai_turn(game_id):
    game_manager = games[game_id]
    
    # Execute AI moves
    result = game_manager.execute_ai_moves(ai_model, device)
    
    return jsonify(result)

@app.route('/api/roll_dice', methods=['POST'])
def roll_dice():
    game_id = session.get('game_id')
    
    if not game_id or game_id not in games:
        return jsonify({'error': 'No active game'}), 400
    
    game_manager = games[game_id]
    
    # Roll dice and get valid moves
    dice, valid_moves_by_piece = game_manager.roll_dice('white')
    
    return jsonify({
        'game_id': game_id,
        'dice': dice,
        'valid_moves_by_piece': valid_moves_by_piece
    })

@app.route('/api/confirm_moves', methods=['POST'])
def confirm_moves():
    game_id = session.get('game_id')
    
    if not game_id or game_id not in games:
        return jsonify({'error': 'No active game'}), 400
    
    # Now let the AI make its turn
    return ai_turn(game_id)

@app.route('/api/undo_moves', methods=['POST'])
def undo_moves():
    game_id = session.get('game_id')
    
    if not game_id or game_id not in games:
        return jsonify({'error': 'No active game'}), 400
    
    game_manager = games[game_id]
    
    # Reset to the state at the beginning of this turn
    result = game_manager.undo_moves()
    
    # Check for errors
    if 'error' in result:
        return jsonify({'error': result['error']}), 400
    
    return jsonify(result)

@app.route('/api/game_state', methods=['GET'])
def get_game_state():
    game_id = session.get('game_id')
    
    if not game_id or game_id not in games:
        return jsonify({'error': 'No active game'}), 400
    
    game_manager = games[game_id]
    game_state = game_manager.get_game_state()
    
    return jsonify(game_state)

@app.route('/api/resign', methods=['POST'])
def resign_game():
    game_id = session.get('game_id')
    
    if not game_id or game_id not in games:
        return jsonify({'error': 'No active game'}), 400
    
    game_manager = games[game_id]
    env = game_manager.env
    
    # Mark the game as resigned
    # In a real implementation, we would update the game state and score
    
    return jsonify({
        'board': env.game.board.tolist(),
        'game_over': True,
        'winner': 'black',  # AI wins when player resigns
        'resigned': True,
        'borne_off': {
            'white': env.game.borne_off_white,
            'black': env.game.borne_off_black
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5858)