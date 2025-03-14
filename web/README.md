# Narde Web Interface

A web-based interface for playing the Narde backgammon variant against a trained Deep Q-Network AI.

## Features

- Interactive drag-and-drop interface
- Visual board representation
- AI opponent powered by a trained DQN model
- Game state tracking and validation
- Support for bearing off pieces

## How to Play

1. Run the web interface:
   ```
   python web_interface.py
   ```

2. Open your browser to http://localhost:5000

3. Click "New Game" to start

4. Game Rules:
   - You play as White, the AI plays as Black
   - Move your pieces counter-clockwise
   - Pieces can only move to empty points or points with your own pieces
   - When all your pieces are in your home quadrant (positions 0-5), you can bear them off
   - First player to bear off all 15 pieces wins
   - The head rule allows only one checker from the starting position per turn (with exceptions)
   - The block rule prevents creating 6+ consecutive points that trap the opponent

## Using the Interface

1. The dice at the top show your current roll
2. Pieces that can move are highlighted
3. To move a piece:
   - Click and hold on a piece to see valid destinations
   - Drag to a valid position or to the bearing off area
   - If you have a second dice, make another move
4. The game displays the AI's moves and updates the board

## Technical Details

This web interface uses:
- Flask as the backend server
- HTML5/CSS/JavaScript for the frontend
- SVG for the game board visualization
- The trained DQN model from `/saved_models/narde_model_final.pt`