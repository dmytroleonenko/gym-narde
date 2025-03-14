/**
 * API Wrapper for communicating with the server
 * 
 * @constructor
 * @param {Comm} comm - Communication system
 */
function ApiWrapper(comm) {
  this.comm = comm;
  this.gameId = null;
}

ApiWrapper.prototype = {
  /**
   * Start a new game
   */
  reqNewGame: function() {
    var self = this;
    
    fetch('/api/new_game', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'}
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
      self.gameId = data.game_id;
      
      console.log('Game started with ID:', data.game_id);
      console.log('Initial board:', data.board);
      
      // Convert player type and create dice object
      var playerType = data.current_player === 'white' ? model.PieceType.WHITE : model.PieceType.BLACK;
      var diceValues = data.dice || [1, 1];
      var diceObj = new model.Dice(diceValues);
      
      // Construct game state
      var gameState = {
        board: data.board,
        turnPlayer: { currentPieceType: playerType },
        turnDice: diceObj,
        isOver: false,
        hasStarted: true,
        pendingActions: []
      };
      
      // Dispatch match start event
      self.comm.dispatch(Comm.Message.EVENT_MATCH_START, {
        game: gameState,
        validMoves: data.valid_moves_by_piece
      });
    })
    .catch(function(error) {
      console.error('Error starting new game:', error);
    });
  },
  
  /**
   * Request to roll dice
   */
  reqRollDice: function() {
    var self = this;
    
    fetch('/api/roll_dice', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ game_id: this.gameId })
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
      // Dispatch dice rolled event
      self.comm.dispatch(Comm.Message.EVENT_DICE_ROLLED, {
        dice: { 
          values: data.dice, 
          movesLeft: data.dice.slice() 
        },
        validMoves: data.valid_moves_by_piece
      });
    })
    .catch(function(error) {
      console.error('Error rolling dice:', error);
    });
  },
  
  /**
   * Request to get valid moves for a piece
   * 
   * @param {number} position - Position of the piece
   */
  reqGetValidMoves: function(position) {
    var self = this;
    
    fetch('/api/get_valid_moves', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ 
        position: position 
      })
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
      // Dispatch valid moves returned event
      self.comm.dispatch('validMovesReceived', {
        position: position,
        validDestinations: data.valid_to_positions
      });
    })
    .catch(function(error) {
      console.error('Error getting valid moves:', error);
    });
  },
  
  /**
   * Request to move a piece
   * 
   * @param {Object} piece - Piece to move
   * @param {number} fromPosition - Starting position
   * @param {number} toPosition - Destination position
   */
  reqMove: function(piece, fromPosition, toPosition) {
    var self = this;
    
    console.log('Requesting move:', fromPosition, '->', toPosition);
    
    fetch('/api/make_move', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        from_position: fromPosition,
        to_position: toPosition
      })
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
      if (data.error) {
        console.error('Move error:', data.error);
        
        // Display error message to user
        if (window.alert) {
          window.alert('Invalid move: ' + data.error);
        }
        
        return;
      }
      
      console.log('Move response:', data);
      
      // Create dice object if needed
      var diceObj = null;
      if (data.dice) {
        diceObj = new model.Dice(data.dice);
        
        // If this is the second move, adjust movesLeft
        if (data.first_move_complete && data.needs_second_move) {
          diceObj.movesLeft = [data.dice[1]];
        } else if (!data.needs_second_move) {
          diceObj.movesLeft = [];
        }
      }
      
      // Process valid moves by piece
      var validMoves = data.valid_moves_by_piece || {};
      
      // Construct updated game state
      var updatedGame = {
        board: data.board,
        turnPlayer: { currentPieceType: data.current_player || 'white' },
        turnDice: diceObj,
        isOver: data.game_over || false,
        hasStarted: true,
        validMoves: validMoves,
        pendingActions: []
      };
      
      // Process the move that was just made
      var moveAction = {
        from: fromPosition,
        to: toPosition === -1 ? 'off' : toPosition,
        piece: piece
      };
      
      // If move was successful, add it to pending actions
      updatedGame.pendingActions.push(moveAction);
      
      // Dispatch move executed event
      self.comm.dispatch(Comm.Message.EVENT_MOVE_EXECUTED, {
        game: updatedGame,
        move: moveAction,
        hasMoreMoves: data.first_move_complete && data.needs_second_move,
        validMoves: validMoves
      });
      
      // If game is over, dispatch match over event
      if (data.game_over) {
        self.comm.dispatch(Comm.Message.EVENT_MATCH_OVER, {
          winner: data.winner,
          score: {
            white: data.borne_off.white,
            black: data.borne_off.black
          }
        });
      }
    })
    .catch(function(error) {
      console.error('Error making move:', error);
    });
  },
  
  /**
   * Request to confirm all pending moves
   */
  reqConfirmMoves: function() {
    var self = this;
    
    fetch('/api/confirm_moves', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ game_id: this.gameId })
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
      // Dispatch moves confirmed event
      self.comm.dispatch(Comm.Message.EVENT_MOVES_CONFIRMED, {
        game: {
          board: data.board,
          turnPlayer: { currentPieceType: data.current_player },
          turnDice: data.dice ? { values: data.dice, movesLeft: data.dice.slice() } : null,
          isOver: data.game_over || false,
          hasStarted: true
        },
        aiMoves: data.ai_moves,
        validMoves: data.valid_moves_by_piece || {}
      });
      
      // If game is over, dispatch match over event
      if (data.game_over) {
        self.comm.dispatch(Comm.Message.EVENT_MATCH_OVER, {
          winner: data.winner,
          score: {
            white: data.borne_off.white,
            black: data.borne_off.black
          }
        });
      }
    })
    .catch(function(error) {
      console.error('Error confirming moves:', error);
    });
  },
  
  /**
   * Request to undo pending moves
   */
  reqUndoMoves: function() {
    var self = this;
    
    fetch('/api/undo_moves', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ game_id: this.gameId })
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
      // Dispatch moves undone event
      self.comm.dispatch(Comm.Message.EVENT_MOVES_UNDONE, {
        game: {
          board: data.board,
          turnPlayer: { currentPieceType: data.current_player },
          turnDice: { values: data.dice, movesLeft: data.dice.slice() },
          isOver: false,
          hasStarted: true
        },
        validMoves: data.valid_moves_by_piece
      });
    })
    .catch(function(error) {
      console.error('Error undoing moves:', error);
    });
  },
  
  /**
   * Request to resign from the game
   */
  reqResignGame: function() {
    var self = this;
    
    fetch('/api/resign', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ game_id: this.gameId })
    })
    .then(function(response) { return response.json(); })
    .then(function(data) {
      // Dispatch match over event
      self.comm.dispatch(Comm.Message.EVENT_MATCH_OVER, {
        winner: data.winner,
        resigned: true,
        score: {
          white: data.borne_off.white,
          black: data.borne_off.black
        }
      });
    })
    .catch(function(error) {
      console.error('Error resigning game:', error);
    });
  }
};

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ApiWrapper;
}
