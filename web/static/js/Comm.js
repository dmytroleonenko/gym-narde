class Comm {
  constructor() {
    this.subscribers = {};
    this.gameState = {
      board: Array(24).fill(0),
      currentPlayer: null,
      dice: [1, 1],
      validMovesByPiece: {},
      firstMoveMade: false,
      selectedPiece: null,
      validDestinations: [],
      gameId: null,
      borneOff: {
        white: 0,
        black: 0
      }
    };
    this.init();
  }

  init() {
    // Listen for global custom events
    window.addEventListener('pointSelected', (e) => this.dispatch('pointSelected', e.detail));
    window.addEventListener('pieceSelected', (e) => this.dispatch('pieceSelected', e.detail));
    window.addEventListener('pieceMoving', (e) => this.dispatch('pieceMoving', e.detail));
    window.addEventListener('pieceDropped', (e) => this.dispatch('pieceDropped', e.detail));
    window.addEventListener('dieClicked', (e) => this.dispatch('dieClicked', e.detail));
    window.addEventListener('actionTriggered', (e) => this.dispatch('actionTriggered', e.detail));
    
    // Set up default handlers for core game mechanics
    this.subscribe('pieceSelected', this.handlePieceSelected.bind(this));
    this.subscribe('pieceDropped', this.handlePieceDropped.bind(this));
    this.subscribe('actionTriggered', this.handleAction.bind(this));
  }

  subscribe(eventName, callback) {
    if (!this.subscribers[eventName]) {
      this.subscribers[eventName] = [];
    }
    this.subscribers[eventName].push(callback);
    return () => {
      this.subscribers[eventName] = this.subscribers[eventName].filter(cb => cb !== callback);
    };
  }

  dispatch(eventName, payload) {
    if (this.subscribers[eventName]) {
      this.subscribers[eventName].forEach(cb => cb(payload));
    }
    console.log(`Comm dispatched ${eventName}`, payload);
  }

  send(msg, payload) {
    console.log(`Sending message: ${msg}`, payload);
    
    // Map message types to API endpoints
    switch(msg) {
      case 'newGame':
        return this.startNewGame();
      
      case 'getValidMoves':
        return this.getValidMoves(payload.position);
      
      case 'makeMove':
        return this.makeMove(payload.fromPosition, payload.toPosition);
      
      case 'rollDice':
        return this.rollDice();
      
      case 'confirmMove':
        return this.confirmMoves();
      
      case 'undoMove':
        return this.undoMoves();
        
      case 'getGameState':
        return this.getGameState();
        
      default:
        console.error(`Unknown message type: ${msg}`);
        return Promise.reject(`Unknown message type: ${msg}`);
    }
  }
  
  // API methods
  async startNewGame() {
    try {
      const response = await fetch('/api/new_game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      
      // Update game state
      this.gameState = {
        board: data.board,
        currentPlayer: data.current_player,
        dice: data.dice,
        validMovesByPiece: data.valid_moves_by_piece || {},
        firstMoveMade: false,
        selectedPiece: null,
        validDestinations: [],
        gameId: data.game_id,
        borneOff: data.borne_off
      };
      
      // Dispatch game state change event
      this.dispatch('gameStateChanged', this.gameState);
      
      return data;
    } catch (error) {
      console.error('Error starting game:', error);
      throw error;
    }
  }
  
  async getValidMoves(position) {
    try {
      const response = await fetch('/api/get_valid_moves', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ position })
      });
      
      const data = await response.json();
      
      // Update game state
      this.gameState.validDestinations = data.valid_to_positions || [];
      this.gameState.selectedPiece = position;
      
      // Dispatch valid moves event
      this.dispatch('validMovesReceived', {
        position,
        validDestinations: data.valid_to_positions || []
      });
      
      return data;
    } catch (error) {
      console.error('Error getting valid moves:', error);
      throw error;
    }
  }
  
  async makeMove(fromPosition, toPosition) {
    try {
      // Track if this is a head move for rule enforcement
      const isHeadMove = fromPosition === 23; // 23 is the white head position
      
      // Check for head rule: only 1 checker from head per turn, except special case
      if (isHeadMove && this.gameState.headMoveMade && !this.isSpecialDoublesCase()) {
        this.dispatch('moveError', { 
          error: 'Only one checker may leave the head position per turn' 
        });
        return { error: 'Head rule violation' };
      }
      
      const response = await fetch('/api/make_move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          from_position: fromPosition,
          to_position: toPosition
        })
      });
      
      const data = await response.json();
      
      if (data.error) {
        // Server reported an error
        this.dispatch('moveError', { error: data.error });
        return data;
      }
      
      // Update game state
      if (data.board) {
        this.gameState.board = data.board;
      }
      
      if (data.borne_off) {
        this.gameState.borneOff = data.borne_off;
      }
      
      // Track head move to enforce head rule
      if (isHeadMove) {
        this.gameState.headMoveMade = true;
      }
      
      if (data.game_over) {
        // Game over logic
        this.gameState.gameOver = true;
        this.gameState.winner = data.winner;
        
        let message = `Game Over! ${data.winner === 'white' ? 'You (White)' : 'AI (Black)'} wins!`;
        
        // Add additional context about scoring (double win or single)
        if (data.winner === 'white' && data.borne_off.black === 0) {
          message += ' Double win (mars) - opponent has no pieces off!';
        } else if (data.winner === 'black' && data.borne_off.white === 0) {
          message += ' Double win (mars) - you have no pieces off!';
        } else {
          message += ' Single win (oin).';
        }
        
        // If AI made the winning move, include info about AI's final moves
        if (data.winner === 'black' && data.ai_moves) {
          this.dispatch('aiMoveComplete', {
            aiMoves: data.ai_moves || [],
            aiDice: data.ai_dice || [],
            aiMovedFromHead: data.ai_moved_from_head,
            isWinningMove: true
          });
        }
        
        this.dispatch('gameOver', {
          winner: data.winner,
          message: message,
          boardState: data.board,
          borneOff: data.borne_off
        });
      } else if (data.first_move_complete && data.needs_second_move) {
        // First move of a two-move sequence
        this.gameState.firstMoveMade = true;
        
        if (data.valid_moves_by_piece) {
          this.gameState.validMovesByPiece = data.valid_moves_by_piece;
        }
        
        // Pass along important game state info, including head move tracking
        this.dispatch('firstMoveComplete', { 
          needsSecondMove: true,
          headMoveMade: isHeadMove
        });
      } else {
        // Move sequence complete, AI's turn
        if (data.current_player === 'white') {
          // AI's turn is complete, back to human player
          this.gameState.currentPlayer = 'white';
          this.gameState.dice = data.dice;
          this.gameState.validMovesByPiece = data.valid_moves_by_piece || {};
          this.gameState.firstMoveMade = false;
          
          // Reset head move tracking for new turn
          this.gameState.headMoveMade = false;
          
          // Include AI moves for UI feedback
          const aiInfo = {
            aiMoves: data.ai_moves || [],
            aiDice: data.ai_dice || [],
            aiMovedFromHead: data.ai_moved_from_head,
            aiHadNoMoves: data.ai_had_no_moves
          };
          
          this.dispatch('aiMoveComplete', aiInfo);
        }
      }
      
      // Update board visualization with all necessary state
      this.dispatch('boardUpdated', {
        ...this.gameState,
        headMoveMade: this.gameState.headMoveMade
      });
      
      return data;
    } catch (error) {
      console.error('Error making move:', error);
      throw error;
    }
  }
  
  // Helper for checking the special first turn double dice case
  isSpecialDoublesCase() {
    // First turn with doubles 3, 4, or 6: can move up to 2 checkers from head
    const dice = this.gameState.dice;
    if (!dice || dice.length !== 2) return false;
    
    // Check if this is doubles 3, 4, or 6
    const isSpecialDice = 
      (dice[0] === 3 && dice[1] === 3) ||
      (dice[0] === 4 && dice[1] === 4) ||
      (dice[0] === 6 && dice[1] === 6);
      
    // Need to know if this is the first turn
    return isSpecialDice && this.gameState.firstTurn === true;
  }
  
  async rollDice() {
    try {
      const response = await fetch('/api/roll_dice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      
      // Reset head move tracking for the new roll
      this.gameState.headMoveMade = false;
      
      // Update game state
      this.gameState.dice = data.dice;
      this.gameState.validMovesByPiece = data.valid_moves_by_piece || {};
      
      // Get the first turn flag from server if available, otherwise assume this is the first turn
      // if we don't have a flag yet
      this.gameState.firstTurn = data.first_turn !== undefined 
        ? data.first_turn 
        : (this.gameState.firstTurn === undefined ? true : this.gameState.firstTurn);
      
      // Check for special doubles case
      const isSpecialDice = (
        (data.dice[0] === 3 && data.dice[1] === 3) ||
        (data.dice[0] === 4 && data.dice[1] === 4) ||
        (data.dice[0] === 6 && data.dice[1] === 6)
      );
      
      // If this is the first turn with special doubles, dispatch a notification
      if (this.gameState.firstTurn && isSpecialDice) {
        this.dispatch('specialDoubles', { 
          dice: data.dice,
          message: 'First turn with doubles 3, 4, or 6: You can move up to 2 checkers from the head position'
        });
      }
      
      // Dispatch dice rolled event with all relevant info
      this.dispatch('diceRolled', {
        dice: data.dice,
        validMovesByPiece: data.valid_moves_by_piece,
        firstTurn: this.gameState.firstTurn,
        isSpecialDoubles: this.gameState.firstTurn && isSpecialDice
      });
      
      return data;
    } catch (error) {
      console.error('Error rolling dice:', error);
      throw error;
    }
  }
  
  async confirmMoves() {
    try {
      const response = await fetch('/api/confirm_moves', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      
      // Process the same way as makeMove since it's AI turn
      if (data.board) {
        this.gameState.board = data.board;
      }
      
      if (data.borne_off) {
        this.gameState.borneOff = data.borne_off;
      }
      
      if (data.game_over) {
        this.gameState.gameOver = true;
        this.gameState.winner = data.winner;
        
        let message = `Game Over! ${data.winner === 'white' ? 'You (White)' : 'AI (Black)'} wins!`;
        
        // Add additional context about scoring (double win or single)
        if (data.winner === 'white' && data.borne_off.black === 0) {
          message += ' Double win (mars) - opponent has no pieces off!';
        } else if (data.winner === 'black' && data.borne_off.white === 0) {
          message += ' Double win (mars) - you have no pieces off!';
        } else {
          message += ' Single win (oin).';
        }
        
        // If AI made the winning move, include info about AI's final moves
        if (data.winner === 'black' && data.ai_moves) {
          this.dispatch('aiMoveComplete', {
            aiMoves: data.ai_moves || [],
            aiDice: data.ai_dice || [],
            aiMovedFromHead: data.ai_moved_from_head,
            isWinningMove: true
          });
        }
        
        this.dispatch('gameOver', {
          winner: data.winner,
          message: message,
          boardState: data.board,
          borneOff: data.borne_off
        });
      } else {
        // AI's turn is complete, back to human player
        this.gameState.currentPlayer = 'white';
        this.gameState.dice = data.dice;
        this.gameState.validMovesByPiece = data.valid_moves_by_piece || {};
        this.gameState.firstMoveMade = false;
        
        // Reset head move tracking for new turn
        this.gameState.headMoveMade = false;
        
        // Include AI moves for UI feedback
        const aiInfo = {
          aiMoves: data.ai_moves || [],
          aiDice: data.ai_dice || [],
          aiMovedFromHead: data.ai_moved_from_head,
          aiHadNoMoves: data.ai_had_no_moves
        };
        
        this.dispatch('aiMoveComplete', aiInfo);
      }
      
      // Dispatch board updated event
      this.dispatch('boardUpdated', {
        ...this.gameState,
        headMoveMade: this.gameState.headMoveMade
      });
      
      return data;
    } catch (error) {
      console.error('Error confirming moves:', error);
      throw error;
    }
  }
  
  async undoMoves() {
    try {
      const response = await fetch('/api/undo_moves', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      
      if (data.error) {
        this.dispatch('undoError', { error: data.error });
        return data;
      }
      
      // Update game state
      this.gameState.board = data.board;
      this.gameState.currentPlayer = data.current_player;
      this.gameState.dice = data.dice;
      this.gameState.validMovesByPiece = data.valid_moves_by_piece || {};
      this.gameState.firstMoveMade = false;
      this.gameState.borneOff = data.borne_off;
      
      // Dispatch board updated event
      this.dispatch('boardUpdated', this.gameState);
      this.dispatch('movesUndone', {});
      
      return data;
    } catch (error) {
      console.error('Error undoing moves:', error);
      throw error;
    }
  }
  
  async getGameState() {
    try {
      const response = await fetch('/api/game_state', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      
      // Update local game state
      this.gameState = {
        board: data.board,
        currentPlayer: data.current_player,
        dice: data.dice,
        validMovesByPiece: data.valid_moves_by_piece || {},
        firstMoveMade: data.first_move_made,
        selectedPiece: this.gameState.selectedPiece,
        validDestinations: this.gameState.validDestinations,
        gameId: this.gameState.gameId,
        borneOff: data.borne_off,
        gameOver: data.game_over,
        winner: data.winner
      };
      
      // Dispatch game state changed event
      this.dispatch('gameStateChanged', this.gameState);
      
      return data;
    } catch (error) {
      console.error('Error getting game state:', error);
      throw error;
    }
  }
  
  // Event handlers
  handlePieceSelected(payload) {
    const { piece, dragStarted } = payload;
    
    // Request valid moves from server
    this.send('getValidMoves', { position: piece.position });
  }
  
  handlePieceDropped(payload) {
    const { piece, x, y } = payload;
    
    // Check if dropping on a valid destination
    let validDrop = false;
    let toPosition;
    
    // Check if dropping on bearing off area
    const bearingOff = document.getElementById('bearing-off');
    if (bearingOff) {
      const rect = bearingOff.getBoundingClientRect();
      if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
        // Check if bearing off is a valid move
        if (this.gameState.validDestinations.includes(-1)) {
          toPosition = -1; // -1 represents bearing off
          validDrop = true;
        }
      }
    }
    
    // Check if dropping on valid point
    if (!validDrop) {
      for (const pos of this.gameState.validDestinations) {
        if (pos === -1) continue; // Skip 'off' for point check
        
        const pointElement = document.getElementById(`point-${pos}`);
        if (pointElement) {
          const rect = pointElement.getBoundingClientRect();
          
          if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
            toPosition = pos;
            validDrop = true;
            break;
          }
        }
      }
    }
    
    // Make the move if valid
    if (validDrop) {
      this.send('makeMove', {
        fromPosition: this.gameState.selectedPiece,
        toPosition: toPosition
      });
    } else {
      // Invalid drop - dispatch event to reset piece
      this.dispatch('invalidDrop', { piece });
    }
  }
  
  handleAction(payload) {
    const { action } = payload;
    
    switch(action) {
      case 'roll':
        this.send('rollDice');
        break;
      
      case 'confirm':
        this.send('confirmMove');
        break;
      
      case 'undo':
        this.send('undoMove');
        break;
      
      case 'newGame':
        this.send('newGame');
        break;
        
      default:
        console.warn(`Unknown action: ${action}`);
    }
  }
}

const comm = new Comm();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = comm;
}

window.comm = comm;
