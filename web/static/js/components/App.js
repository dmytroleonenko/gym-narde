/**
 * Main application entry point
 * 
 * @constructor
 */
function App() {
  /**
   * Configuration
   * @type {Object}
   */
  this._config = {
    containerID: 'backgammon',
    serverURL: '', // Will be determined from window.location
    isAIGame: false
  };
  
  /**
   * Current player
   * @type {Object}
   */
  this.player = {
    id: null,
    name: 'You',
    currentPieceType: model.PieceType.WHITE
  };
  
  /**
   * Board UI
   * @type {SimpleBoardUI}
   */
  this.ui = null;
  
  /**
   * Current view
   * @type {string}
   */
  this.currentView = 'index';
  
  /**
   * API Wrapper
   * @type {ApiWrapper}
   */
  this.api = null;
  
  /**
   * Flag for waiting state
   * @type {boolean}
   */
  this.isWaiting = false;
  
  /**
   * Flag for challenging state
   * @type {boolean}
   */
  this.isChallenging = false;
}

App.prototype = {
  /**
   * Get name of rule selected by player
   * @returns {string} - Name of selected rule.
   */
  getSelectedRuleName: function() {
    var selected = $('#rule-selector label.active input').val();
    return selected || 'LongNardy';
  },
  
  /**
   * Set the current view
   * @param {string} view - View name
   */
  setCurrentView: function(view) {
    this.currentView = view;
  },
  
  /**
   * Set the waiting state
   * @param {boolean} isWaiting - Waiting state
   */
  setIsWaiting: function(isWaiting) {
    this.isWaiting = isWaiting;
    this.updateView();
  },
  
  /**
   * Set the challenging state
   * @param {boolean} isChallenging - Challenging state
   */
  setIsChallenging: function(isChallenging) {
    this.isChallenging = isChallenging;
    $('#waiting-overlay .challenge').toggle(isChallenging);
    this.updateView();
  },
  
  /**
   * Update the view based on current state
   */
  updateView: function() {
    console.log('Updating view - current view:', this.currentView, 'isWaiting:', this.isWaiting);
    
    // Toggle appropriate view elements
    $('#waiting-overlay').toggle(this.isWaiting);
    $('#game-view').toggle(this.currentView === 'game');
    $('#index-view').toggle(this.currentView === 'index');
    $('#github-ribbon').toggle(this.currentView === 'index');
    
    // Update game status text if in game view
    if (this.currentView === 'game' && this.match) {
      var statusText = '';
      var game = this.match.currentGame;
      
      if (game.isOver) {
        // Game is over
        statusText = 'Game Over - ' + 
                     (this.match.winner === this.player.currentPieceType ? 'You won!' : 'You lost!');
      } else if (game.turnPlayer.currentPieceType === this.player.currentPieceType) {
        // Player's turn
        if (game.turnDice) {
          // Dice rolled
          if (game.turnDice.movesLeft && game.turnDice.movesLeft.length > 0) {
            statusText = 'Your turn - make a move';
          } else {
            statusText = 'Your turn - confirm or undo moves';
          }
        } else {
          statusText = 'Your turn - roll dice';
        }
      } else {
        // Opponent's turn
        statusText = "Opponent's turn";
      }
      
      // Update navbar match state
      $('#match-state').text(statusText);
    }
    
    console.log('View updated');
  },
  
  /**
   * Initialize the rule selector
   */
  initRuleSelector: function() {
    var self = this;
    
    // Init rule selector with only Long Nardy
    var selector = $('#rule-selector');
    selector.empty();
    
    var ruleName = 'LongNardy';
    var ruleTitle = 'Long Nardy';
    var isActive = true;
    var isChecked = true;
    
    var item = $('#tmpl-rule-selector-item').html();
    item = item.replace('{{name}}', ruleName);
    item = item.replace('{{title}}', ruleTitle);
    item = item.replace('{{active}}', isActive ? 'active' : '');
    item = item.replace('{{checked}}', isChecked ? 'checked' : '');
    selector.append($(item));
  },
  
  /**
   * Initialize the application
   * @param {Object} config - Configuration object
   */
  init: function(config) {
    var self = this;
    
    // Merge configurations
    if (config) {
      Object.keys(config).forEach(function(key) {
        self._config[key] = config[key];
      });
    }
    
    // Initialize rule selector
    this.initRuleSelector();
    
    // Initialize game result overlay click handler
    $('#game-result-overlay').click(function() {
      $('#game-result-overlay').hide();
    });
    
    // Initialize API wrapper
    this.api = new ApiWrapper(comm);
    
    // Initialize UI
    this.ui = new SimpleBoardUI(this);
    
    // Subscribe to events
    this.subscribeToEvents();
    
    // Set up button event handlers
    this.setupButtons();
    
    // Update view
    this.updateView();
    
    // Initialize clipboard library for copying challenge links
    if (typeof ClipboardJS !== 'undefined') {
      new ClipboardJS('.btn-copy');
    }
    
    // Handle window resize
    $(window).resize(function() {
      if (self.ui) {
        self.ui.resize();
      }
    });
  },
  
  /**
   * Subscribe to communication events
   */
  subscribeToEvents: function() {
    var self = this;
    
    console.log('Subscribing to events');
    
    // Game start event
    comm.subscribe(Comm.Message.EVENT_MATCH_START, function(data) {
      console.log('EVENT_MATCH_START received:', data);
      
      self.setIsWaiting(false);
      self.setCurrentView('game');
      
      // Initialize match object
      self.match = {
        currentGame: data.game,
        isStarted: true,
        currentGameNumber: 1,
        gameCount: 1,
        score: { white: 0, black: 0 },
        pointsToWin: 1
      };
      
      // Initialize board
      if (self.ui) {
        self.ui.match = self.match;
        self.ui.resetBoard(self.match, null);
      }
      
      self.updateView();
      console.log('Match started, board reset');
    });
    
    // Match over event
    comm.subscribe(Comm.Message.EVENT_MATCH_OVER, function(data) {
      console.log('EVENT_MATCH_OVER received:', data);
      
      if (self.match) {
        self.match.isOver = true;
        self.match.currentGame.isOver = true;
        self.match.winner = data.winner;
        
        // Update score
        if (data.score) {
          self.match.score = data.score;
        }
        
        // Show game result
        var winnerType = data.winner;
        var playerWon = winnerType === self.player.currentPieceType;
        var message = playerWon ? 'You won!' : 'You lost!';
        var color = playerWon ? '#37BC9B' : '#DA4453';
        var matchState = data.resigned ? 'Opponent resigned' : 'Game over';
        
        if (self.ui) {
          self.ui.showGameResult(message, matchState, color);
        }
        
        self.updateView();
        
        // Return to index view after delay
        setTimeout(function() {
          self.setCurrentView('index');
          self.updateView();
        }, 5000);
      }
    });
    
    // Move executed event
    comm.subscribe(Comm.Message.EVENT_MOVE_EXECUTED, function(data) {
      console.log('EVENT_MOVE_EXECUTED received:', data);
      
      if (self.match) {
        // Update game state
        self.match.currentGame = data.game;
        
        // Update board
        if (self.ui) {
          self.ui.updateBoard();
        }
        
        self.updateView();
      }
    });
    
    // Moves confirmed event
    comm.subscribe(Comm.Message.EVENT_MOVES_CONFIRMED, function(data) {
      console.log('EVENT_MOVES_CONFIRMED received:', data);
      
      if (self.match) {
        // Update game state
        self.match.currentGame = data.game;
        
        // Update board
        if (self.ui) {
          self.ui.updateBoard();
        }
        
        self.updateView();
      }
    });
    
    // Moves undone event
    comm.subscribe(Comm.Message.EVENT_MOVES_UNDONE, function(data) {
      console.log('EVENT_MOVES_UNDONE received:', data);
      
      if (self.match) {
        // Update game state
        self.match.currentGame = data.game;
        
        // Update board
        if (self.ui) {
          self.ui.updateBoard();
          self.ui.notifyUndo();
        }
        
        self.updateView();
      }
    });
    
    // Dice rolled event
    comm.subscribe(Comm.Message.EVENT_DICE_ROLLED, function(data) {
      console.log('EVENT_DICE_ROLLED received:', data);
      
      if (self.match) {
        // Update dice in game state
        self.match.currentGame.turnDice = data.dice;
        
        // If valid moves were provided, store them
        if (data.validMoves) {
          self.match.currentGame.validMoves = data.validMoves;
        }
        
        // Update board
        if (self.ui) {
          self.ui.updateControls();
        }
        
        self.updateView();
      }
    });
    
    console.log('Event subscriptions complete');
  },
  
  /**
   * Set up button event handlers
   */
  setupButtons: function() {
    var self = this;
    
    // Play against AI button
    $('#btn-play-ai').click(function() {
      self.setIsChallenging(false);
      self.setIsWaiting(true);
      self._config.isAIGame = true;
      
      // Start a new game against AI
      self.api.reqNewGame();
    });
    
    // Challenge friend button
    $('#btn-challenge-friend').click(function() {
      alert('Multiplayer mode is not implemented yet. Please use the "Play against AI" button.');
      
      // In a real implementation, this would create a match and show challenge link
      /*
      self._config.isAIGame = false;
      self.api.reqCreateMatch(self.getSelectedRuleName(), function(reply) {
        $('#challenge-link').val(window.location.origin + '?join=' + reply.matchID);
        self.setIsChallenging(true);
      });
      */
    });
  },
  
  /**
   * Request to roll dice
   */
  reqRollDice: function() {
    this.api.reqRollDice();
  },
  
  /**
   * Request to move a piece
   * @param {Object} piece - Piece to move
   * @param {number} steps - Number of steps to move
   */
  reqMove: function(piece, steps) {
    // Calculate destination position
    var from = parseInt($(piece).parent().data('position'));
    var to = from - steps;
    
    if (to < 0) {
      to = 'off'; // Bearing off
    }
    
    this.api.reqMove(piece, from, to);
  },
  
  /**
   * Request to move piece up (Long Nardy specific)
   * @param {Object} piece - Piece to move up
   * @param {number} height - Height to move to
   */
  reqUp: function(piece, height) {
    // This is a Long Nardy specific move that just changes the piece's visual height
    // Note: Not fully implemented yet
    console.log('Up move requested:', piece, height);
  },
  
  /**
   * Request to confirm moves
   */
  reqConfirmMoves: function() {
    this.api.reqConfirmMoves();
  },
  
  /**
   * Request to undo moves
   */
  reqUndoMoves: function() {
    this.api.reqUndoMoves();
  },
  
  /**
   * Request to resign from game
   */
  reqResignGame: function() {
    this.api.reqResignGame();
  },
  
  /**
   * Update the UI after window resize
   */
  resizeUI: function() {
    if (this.ui) {
      this.ui.resize();
    }
  },
  
  /**
   * Start a game against the AI
   */
  playAgainstAI: function() {
    var self = this;
    
    // Show waiting overlay
    this.setIsWaiting(true);
    this._config.isAIGame = true;
    
    // Call the API to start a new game
    this.api.reqNewGame();
  }
};

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = App;
}