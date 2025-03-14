class App {
  constructor(containerId) {
    this.containerId = containerId;
    this.container = document.getElementById(containerId);
    this.gameState = {
      board: Array(24).fill(0),
      pieces: [],
      validMovesByPiece: {},
      selectedPiece: null,
      validDestinations: []
    };
    this.headMoveUsed = false;
    this.init();
    this.bindComm();
  }

  init() {
    // Create main game container
    this.gameContainer = document.createElement('div');
    this.gameContainer.className = 'game-container';
    this.container.appendChild(this.gameContainer);
    
    // Create board container
    this.boardContainerEl = document.createElement('div');
    this.boardContainerEl.className = 'board-container';
    this.boardContainerEl.id = 'board-container';
    this.gameContainer.appendChild(this.boardContainerEl);
    
    // Create action panel container
    this.actionPanelEl = document.createElement('div');
    this.actionPanelEl.className = 'action-panel-container';
    this.gameContainer.appendChild(this.actionPanelEl);
    
    // Create dice container
    this.diceContainerEl = document.createElement('div');
    this.diceContainerEl.className = 'dice-container-wrapper';
    this.gameContainer.appendChild(this.diceContainerEl);
    
    // Initialize components
    this.boardContainer = new BoardContainer('board-container');
    this.actionPanel = new ActionPanel(this.actionPanelEl);
    this.diceComponent = new DiceComponent(this.diceContainerEl);
    
    // Pieces will be created dynamically based on board state
    this.pieces = [];
    
    // Listen for special doubles event
    window.addEventListener('specialDoublesDetected', this.handleSpecialDoublesDetected.bind(this));
  }
  
  handleSpecialDoublesDetected(event) {
    // Show the special doubles notification in the BoardContainer
    if (this.boardContainer) {
      this.boardContainer.showSpecialDoublesNotification(true);
    }
  }

  bindComm() {
    // Subscribe to communication events
    
    // Board/point events
    comm.subscribe('pointSelected', (detail) => {
      console.log('App handling pointSelected:', detail);
      const { index } = detail;
      
      // If we have a selected piece, attempt to move it to this point
      if (this.gameState.selectedPiece !== null) {
        this.attemptMove(this.gameState.selectedPiece, index);
      }
    });

    // Piece events
    comm.subscribe('pieceSelected', (detail) => {
      console.log('App handling pieceSelected:', detail);
      const { piece, dragStarted } = detail;
  
      // Check for head rule (position 23 for white)
      if (piece.position === 23) {
        // If we already made a head move and this isn't a special case, show notification
        if (this.headMoveUsed && !this.isSpecialDoublesCase()) {
          this.boardContainer.showHeadRuleNotification(true);
          return;
        }
      }
  
      // Set the selected piece
      this.gameState.selectedPiece = piece.position;
  
      // Get valid moves for this piece
      comm.send('getValidMoves', { position: piece.position });
    });
    
    comm.subscribe('pieceMoving', (detail) => {
      // Handle piece being dragged (visual feedback only)
      console.log('App handling pieceMoving:', detail);
    });
    
    comm.subscribe('pieceDropped', (detail) => {
      console.log('App handling pieceDropped:', detail);
      // The Comm class will handle the drop logic directly by checking drop position
    });

    // Bearing off events
    comm.subscribe('bearingOffSelected', (detail) => {
      console.log('App handling bearingOffSelected:', detail);
      
      // If we have a selected piece, attempt to bear it off
      if (this.gameState.selectedPiece !== null) {
        this.attemptMove(this.gameState.selectedPiece, -1); // -1 represents bearing off
      }
    });

    // Die events
    comm.subscribe('dieClicked', (detail) => {
      console.log('App handling dieClicked:', detail);
      // We could use this to filter valid moves by die value if needed
    });

    // Action events
    comm.subscribe('actionTriggered', (detail) => {
      console.log('App handling actionTriggered:', detail);
      
      // Handle specific actions
      switch(detail.action) {
        case 'roll':
          // Clear selected piece when rolling dice
          this.gameState.selectedPiece = null;
          this.gameState.validDestinations = [];
          this.gameState.headMoveMade = false; // Reset head move tracking on new roll
          this.headMoveUsed = false;
          
          // Reset any rule visualizations
          this.resetRuleVisualizations();
          break;
          
        case 'new-game':
          // Reset game state
          this.gameState = {
            board: Array(24).fill(0),
            pieces: [],
            validMovesByPiece: {},
            selectedPiece: null,
            validDestinations: [],
            headMoveMade: false
          };
          this.headMoveUsed = false;
          
          // Reset any rule visualizations
          this.resetRuleVisualizations();
          break;
          
        case 'undo':
          // Clear selected piece when undoing moves
          this.gameState.selectedPiece = null;
          this.gameState.validDestinations = [];
          this.gameState.headMoveMade = false; // Reset head move tracking on undo
          this.headMoveUsed = false;
          
          // Reset any rule visualizations
          this.resetRuleVisualizations();
          break;
      }
      
      // Forward action to comm.send
      comm.send(detail.action, {});
    });
    
    // Game state events
    comm.subscribe('gameStateChanged', (data) => {
      console.log('App handling gameStateChanged:', data);
      this.gameState = {
        ...data,
        headMoveMade: false, // Reset head move tracking on new game state
        validDestinations: this.gameState.validDestinations // Preserve destinations
      };
      this.renderBoard();
      
      // Check for special doubles case on first turn
      this.checkForSpecialDoublesCase(data.dice);
      
      // Highlight home areas for the current player
      this.boardContainer.highlightHomeAreas(data.currentPlayer);
    });
    
    comm.subscribe('validMovesReceived', (data) => {
      console.log('App handling validMovesReceived:', data);
      const { position, validDestinations } = data;
      
      // Update valid destinations
      this.gameState.validDestinations = validDestinations;
      
      // Highlight valid destinations on board
      this.highlightValidDestinations(validDestinations);
    });
    
    comm.subscribe('boardUpdated', (data) => {
      console.log('App handling boardUpdated:', data);
      
      // Preserve headMoveMade state if this is a partial board update
      const headMoveMade = data.headMoveMade !== undefined ? 
        data.headMoveMade : 
        (this.gameState.headMoveMade || false);
        
      this.gameState = {
        ...data,
        headMoveMade,
        selectedPiece: null,
        validDestinations: []  // Clear valid destinations to avoid stale moves
      };
      
      this.renderBoard();
      
      // Update block rule visualizations after board update
      this.detectAndVisualizeBlocks();
      
      // Check for AI head moves and visualize if needed
      if (data.ai_moved_from_head) {
        // Show notification that AI moved from head position
        if (this.boardContainer) {
          this.boardContainer.headRuleNotification.textContent = 'AI moved a checker from the head position';
          this.boardContainer.showHeadRuleNotification(true);
        }
      }
      
      // Visualize AI moves with arrows if available
      if (data.aiMoves && data.aiMoves.length > 0 && this.boardContainer) {
        this.boardContainer.visualizeAIMoves(data.aiMoves, data.isWinningMove);
      }
      
      // If AI had no valid moves, show a message
      if (data.aiHadNoMoves) {
        const notification = document.createElement('div');
        notification.className = 'ai-no-moves-notification';
        notification.textContent = 'AI had no valid moves. Your turn!';
        notification.style.position = 'absolute';
        notification.style.top = '50%';
        notification.style.left = '50%';
        notification.style.transform = 'translate(-50%, -50%)';
        notification.style.backgroundColor = '#2196F3';
        notification.style.color = 'white';
        notification.style.padding = '10px 20px';
        notification.style.borderRadius = '5px';
        notification.style.zIndex = '100';
        
        this.boardContainer.boardEl.appendChild(notification);
        
        // Auto-remove after 2 seconds
        setTimeout(() => {
          notification.remove();
        }, 2000);
      }
    });
    
    comm.subscribe('invalidDrop', (data) => {
      console.log('App handling invalidDrop:', data);
      // Reset the board to show the piece in its original position
      this.renderBoard();
    });
    
    comm.subscribe('moveError', (data) => {
      console.log('App handling moveError:', data);
      
      // If the error is related to the head rule
      if (data.error && data.error.includes('head position')) {
        if (this.boardContainer) {
          this.boardContainer.showHeadRuleNotification(true);
        }
      } 
      // If the error is related to the block rule
      else if (data.error && data.error.includes('block')) {
        if (this.boardContainer) {
          this.boardContainer.showBlockRuleWarning(true);
        }
      }
      
      // Reset the board to show the piece in its original position
      this.renderBoard();
    });
    
    comm.subscribe('gameOver', (data) => {
      console.log('App handling gameOver:', data);
      
      // Update board one last time to ensure final state is shown
      if (data.boardState) {
        this.gameState.board = data.boardState;
        this.renderBoard();
      }
      
      // Create a nicer game over display instead of alert
      const gameOverOverlay = document.createElement('div');
      gameOverOverlay.className = 'game-over-overlay';
      
      const gameOverContent = document.createElement('div');
      gameOverContent.className = 'game-over-content';
      
      // Add header with winner info
      const header = document.createElement('h2');
      header.textContent = data.winner === 'white' ? 'You Win!' : 'AI Wins!';
      header.className = data.winner;
      gameOverContent.appendChild(header);
      
      // Add game stats
      const stats = document.createElement('div');
      stats.className = 'game-over-stats';
      
      // Show borne off pieces count
      if (data.borneOff) {
        const borneOffInfo = document.createElement('p');
        borneOffInfo.innerHTML = `White pieces borne off: <strong>${data.borneOff.white}</strong><br>` +
                                 `Black pieces borne off: <strong>${data.borneOff.black}</strong>`;
        stats.appendChild(borneOffInfo);
      }
      
      // Win type
      const winType = document.createElement('p');
      if (data.winner === 'white' && data.borneOff?.black === 0) {
        winType.innerHTML = '<strong>Double win (mars)</strong> - opponent has no pieces off!';
      } else if (data.winner === 'black' && data.borneOff?.white === 0) {
        winType.innerHTML = '<strong>Double win (mars)</strong> - you have no pieces off!';
      } else {
        winType.innerHTML = '<strong>Single win (oin)</strong>';
      }
      stats.appendChild(winType);
      
      gameOverContent.appendChild(stats);
      
      // Add new game button
      const newGameBtn = document.createElement('button');
      newGameBtn.className = 'btn btn-primary';
      newGameBtn.textContent = 'Play Again';
      newGameBtn.addEventListener('click', () => {
        // Remove the overlay
        gameOverOverlay.remove();
        
        // Start a new game
        comm.send('newGame');
      });
      gameOverContent.appendChild(newGameBtn);
      
      gameOverOverlay.appendChild(gameOverContent);
      document.body.appendChild(gameOverOverlay);
    });
    
    comm.subscribe('firstMoveComplete', (data) => {
      console.log('App handling firstMoveComplete:', data);
      
      // Update tracking for head rule
      if (this.gameState.selectedPiece === 23) {
        this.gameState.headMoveMade = true;
      }
      
      // Clear selected piece after first move
      this.gameState.selectedPiece = null;
      this.gameState.validDestinations = [];
    });
  }
  
  // New methods for Long Nardy rules
  
  resetRuleVisualizations() {
    if (this.boardContainer) {
      // Hide all rule notifications
      this.boardContainer.showHeadRuleNotification(false);
      this.boardContainer.showSpecialDoublesNotification(false);
      this.boardContainer.showBlockRuleWarning(false);
    }
  }
  
  isSpecialDoublesCase() {
    // Check if we're on the first turn with doubles 3, 4, or 6
    if (!this.gameState.dice || this.gameState.dice.length !== 2) return false;
    
    const isDiceSpecial = (
      (this.gameState.dice[0] === 3 && this.gameState.dice[1] === 3) ||
      (this.gameState.dice[0] === 4 && this.gameState.dice[1] === 4) ||
      (this.gameState.dice[0] === 6 && this.gameState.dice[1] === 6)
    );
    
    return isDiceSpecial && this.gameState.firstTurn && !this.gameState.headMoveMadeSecond;
  }
  
  checkForSpecialDoublesCase(dice) {
    if (!dice || dice.length !== 2) return;
    
    const isDiceSpecial = (
      (dice[0] === 3 && dice[1] === 3) ||
      (dice[0] === 4 && dice[1] === 4) ||
      (dice[0] === 6 && dice[1] === 6)
    );
    
    if (isDiceSpecial && this.gameState.firstTurn) {
      if (this.boardContainer) {
        this.boardContainer.showSpecialDoublesNotification(true);
      }
    }
  }
  
  detectAndVisualizeBlocks() {
    if (!this.gameState.board || !this.boardContainer) return;
    
    const board = this.gameState.board;
    const blocks = [];
    
    // Find contiguous blocks of 5 or more pieces (to give early warning)
    let currentBlockStart = -1;
    let currentBlockLength = 0;
    
    for (let i = 0; i < 24; i++) {
      // For white pieces (positive numbers)
      if (board[i] > 0) {
        if (currentBlockStart === -1) {
          currentBlockStart = i;
          currentBlockLength = 1;
        } else {
          currentBlockLength++;
        }
      } else {
        // End of a block
        if (currentBlockLength >= 5) {
          blocks.push({
            start: currentBlockStart,
            end: i - 1,
            width: currentBlockLength
          });
        }
        currentBlockStart = -1;
        currentBlockLength = 0;
      }
    }
    
    // Check for a block that ends at the edge of the board
    if (currentBlockLength >= 5) {
      blocks.push({
        start: currentBlockStart,
        end: 23,
        width: currentBlockLength
      });
    }
    
    // Update block visualizations
    this.boardContainer.updateBlockRuleIndicators(blocks);
    
    // If there's a block of exactly 5 pieces, show a warning about block rule
    const hasBlockOfFive = blocks.some(block => block.width === 5);
    if (hasBlockOfFive) {
      // Create a timeout so it doesn't show immediately but after a short delay
      setTimeout(() => {
        this.boardContainer.blockRuleWarning.textContent = 
          'Notice: You have 5 pieces in a row. Adding one more would create a block of 6, which is restricted by Rule 8.';
        this.boardContainer.showBlockRuleWarning(true);
      }, 500);
    }
  }
  
  // Attempt to move a piece
  attemptMove(fromPosition, toPosition) {
    // Check if this is a valid move
    if (this.gameState.validDestinations.includes(toPosition)) {
      if (fromPosition === 23) {
        this.headMoveUsed = true;
      }
      // Send move to server
      comm.send('makeMove', {
        fromPosition: fromPosition,
        toPosition: toPosition
      });
    } else {
      console.log(`Invalid move from ${fromPosition} to ${toPosition}`);
    }
  }
  
  // Highlight valid destinations on the board
  highlightValidDestinations(destinations) {
    // Clear previous highlights
    document.querySelectorAll('.point').forEach(point => {
      point.classList.remove('valid-destination');
    });
    
    // Remove bearing off highlight
    const bearingOff = document.getElementById('bearing-off');
    if (bearingOff) {
      bearingOff.classList.remove('valid-destination');
    }
    
    // Add new highlights
    destinations.forEach(pos => {
      if (pos === -1) {
        // Highlight bearing off area
        if (bearingOff) {
          bearingOff.classList.add('valid-destination');
        }
      } else {
        // Highlight point
        const pointElement = document.getElementById(`point-${pos}`);
        if (pointElement) {
          pointElement.classList.add('valid-destination');
        }
      }
    });
  }
  
  // Render the board based on current game state
  renderBoard() {
    // Clear previous pieces
    this.pieces.forEach(piece => {
      piece.el.remove();
    });
    this.pieces = [];
    
    // Clear highlights
    this.highlightValidDestinations([]);
    
    // Create new pieces based on board state
    const { board } = this.gameState;
    const piecesContainer = document.getElementById('pieces-container') || document.createElement('div');
    piecesContainer.id = 'pieces-container';
    piecesContainer.innerHTML = '';
    
    if (!document.getElementById('pieces-container')) {
      this.boardContainerEl.appendChild(piecesContainer);
    }
    
    // Loop through all positions and create pieces
    for (let position = 0; position < 24; position++) {
      const count = board[position];
      if (count !== 0) {
        // Create a piece at this position
        const isWhite = count > 0;
        const absCount = Math.abs(count);
        
        // Create piece data
        const pieceData = {
          id: `piece-${position}`,
          position: position,
          color: isWhite ? 'white' : 'black',
          count: absCount,
          canMove: isWhite && position in this.gameState.validMovesByPiece
        };
        
        // Create piece component
        const piece = new PieceComponent(pieceData, piecesContainer);
        this.pieces.push(piece);
      }
    }
  }
}

// Export if using modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = App;
}

window.App = App;
