class ActionPanel {
  constructor(container) {
    this.container = container;
    this.gameStarted = false;
    this.moveMade = false;
    this.init();
  }

  init() {
    this.el = document.createElement('div');
    this.el.className = 'action-panel';

    // Create action buttons
    this.buttons = {};
    const actions = [
      { id: 'roll', label: 'Roll', primary: true },
      { id: 'confirm', label: 'Confirm', primary: true },
      { id: 'undo', label: 'Undo', primary: false },
      { id: 'new-game', label: 'New Game', primary: true }
    ];

    actions.forEach(act => {
      const btn = document.createElement('button');
      btn.id = `btn-${act.id}`;
      btn.className = `action-button ${act.primary ? 'primary' : 'secondary'}`;
      btn.textContent = act.label;
      btn.addEventListener('click', () => this.handleAction(act.id));
      this.el.appendChild(btn);
      this.buttons[act.id] = btn;
    });

    // Create status message area
    this.statusEl = document.createElement('div');
    this.statusEl.className = 'action-status';
    this.el.appendChild(this.statusEl);

    // Add event listeners
    window.addEventListener('boardUpdated', this.handleBoardUpdated.bind(this));
    window.addEventListener('gameStateChanged', this.handleGameStateChanged.bind(this));
    window.addEventListener('firstMoveComplete', this.handleFirstMoveComplete.bind(this));
    window.addEventListener('gameOver', this.handleGameOver.bind(this));
    window.addEventListener('aiMoveComplete', this.handleAiMoveComplete.bind(this));
    window.addEventListener('diceRolled', this.handleDiceRolled.bind(this));
    window.addEventListener('moveError', this.handleMoveError.bind(this));

    this.container.appendChild(this.el);
    
    // Initial button state
    this.updateButtonVisibility();
  }

  handleAction(actionId) {
    console.log(`Action ${actionId} triggered`);
    
    switch(actionId) {
      case 'roll':
        this.moveMade = false;
        break;
        
      case 'confirm':
        // Nothing to do specifically here
        break;
        
      case 'undo':
        this.moveMade = false;
        break;
        
      case 'new-game':
        this.gameStarted = true;
        this.moveMade = false;
        break;
    }
    
    const event = new CustomEvent('actionTriggered', { detail: { action: actionId } });
    window.dispatchEvent(event);
    
    // Update buttons
    this.updateButtonVisibility();
  }
  
  updateButtonVisibility() {
    // Hide all buttons first
    for (const key in this.buttons) {
      this.buttons[key].style.display = 'none';
    }
    
    if (!this.gameStarted) {
      // Only show New Game button
      this.buttons['new-game'].style.display = 'block';
      return;
    }
    
    // Game started
    if (this.moveMade) {
      // Show confirm and undo after move
      this.buttons['confirm'].style.display = 'block';
      this.buttons['undo'].style.display = 'block';
    } else {
      // Show roll when no move made yet
      this.buttons['roll'].style.display = 'block';
    }
    
    // Always show New Game
    this.buttons['new-game'].style.display = 'block';
  }
  
  setStatusMessage(message) {
    this.statusEl.textContent = message;
  }
  
  // Event handlers
  handleBoardUpdated(event) {
    if (event.detail) {
      const gameState = event.detail;
      
      // Update based on game state
      if (gameState.firstMoveMade) {
        this.setStatusMessage('Make your second move');
      } else if (gameState.currentPlayer === 'white') {
        if (!this.moveMade) {
          this.setStatusMessage('Your turn - Roll dice');
        }
      } else {
        this.setStatusMessage('AI is thinking...');
      }
      
      this.updateButtonVisibility();
    }
  }
  
  handleGameStateChanged(event) {
    if (event.detail) {
      this.gameStarted = event.detail.gameId !== null;
      this.updateButtonVisibility();
    }
  }
  
  handleFirstMoveComplete(event) {
    this.moveMade = true;
    this.setStatusMessage('Select your second move');
    this.updateButtonVisibility();
  }
  
  handleGameOver(event) {
    if (event.detail) {
      this.gameStarted = false;
      this.setStatusMessage(`Game Over! ${event.detail.winner === 'white' ? 'You (White)' : 'AI (Black)'} wins!`);
      this.updateButtonVisibility();
    }
  }
  
  handleAiMoveComplete(event) {
    this.moveMade = false;
    
    // Display AI's move info
    if (event.detail) {
      const { aiMoves, aiDice, aiMovedFromHead, aiHadNoMoves } = event.detail;
      
      if (aiHadNoMoves) {
        this.setStatusMessage('AI had no valid moves. Your turn - Roll dice');
      } else {
        let moveText = 'AI moved: ';
        aiMoves.forEach((move, i) => {
          moveText += `${move.from} → ${move.to === -1 ? 'Off' : move.to}`;
          if (i < aiMoves.length - 1) moveText += ', ';
        });
        this.setStatusMessage(moveText);
      }
    } else {
      this.setStatusMessage('Your turn - Roll dice');
    }
    
    this.updateButtonVisibility();
  }
  
  handleDiceRolled(event) {
    if (event.detail && event.detail.dice) {
      this.setStatusMessage(`Your turn - Dice: ${event.detail.dice.join(', ')}`);
    }
  }
  
  handleMoveError(event) {
    if (event.detail && event.detail.error) {
      this.setStatusMessage(`Error: ${event.detail.error}`);
    }
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = ActionPanel;
}

window.ActionPanel = ActionPanel;
