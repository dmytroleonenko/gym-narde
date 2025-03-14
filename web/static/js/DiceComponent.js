class DiceComponent {
  constructor(container) {
    this.container = container;
    this.dice = [0, 0];
    this.usedDice = [false, false];
    this.isFirstTurn = true;
    this.isDoubles = false;
    this.init();
  }

  init() {
    this.el = document.createElement('div');
    this.el.className = 'dice-component';

    // Create dice elements
    this.diceEls = [];
    for (let i = 0; i < 2; i++) {
      const dieEl = document.createElement('div');
      dieEl.className = 'die';
      dieEl.textContent = this.dice[i];
      this.el.appendChild(dieEl);
      this.diceEls.push(dieEl);
    }

    // Create optional message element for feedback
    this.messageEl = document.createElement('div');
    this.messageEl.className = 'dice-message';
    this.el.appendChild(this.messageEl);

    this.bindEvents();
    this.container.appendChild(this.el);
    
    // Listen for events
    window.addEventListener('firstMoveComplete', this.handleFirstMoveComplete.bind(this));
    window.addEventListener('boardUpdated', this.handleBoardUpdated.bind(this));
    window.addEventListener('diceRolled', this.handleDiceRolled.bind(this));
  }

  bindEvents() {
    // Dice click rotates dice values
    this.diceEls.forEach((dieEl, index) => {
      dieEl.addEventListener('click', () => {
        this.onDieClick(index);
      });
    });
  }

  onDieClick(index) {
    // Skip if this die is already used
    if (this.usedDice[index]) {
      console.log(`Die ${index} already used`);
      return;
    }
    
    console.log(`Die ${index} clicked, value: ${this.dice[index]}`);
    
    // Mark this die as being selected for current move
    this.diceEls[index].classList.add('selected');
    
    // Dispatch event for die clicked with value
    const event = new CustomEvent('dieClicked', { 
      detail: { 
        index,
        value: this.dice[index],
        dice: this.dice
      } 
    });
    window.dispatchEvent(event);
  }

  updateDice(values) {
    this.dice = values;
    this.isDoubles = values[0] === values[1];
    
    // Reset used state
    this.usedDice = [false, false];
    this.diceEls.forEach((dieEl, i) => {
      dieEl.textContent = values[i];
      dieEl.className = 'die'; // Reset classes
      
      // For doubles, add special class
      if (this.isDoubles) {
        dieEl.classList.add('doubles');
      }
    });
    
    // Clear any message
    this.messageEl.textContent = '';
    this.messageEl.style.display = 'none';
  }
  
  // Mark a die as used after a move
  markDieUsed(dieValue) {
    let dieIndex = -1;
    
    // Find which die matches this value
    for (let i = 0; i < this.dice.length; i++) {
      if (this.dice[i] === dieValue && !this.usedDice[i]) {
        dieIndex = i;
        break;
      }
    }
    
    if (dieIndex === -1) {
      console.warn(`Could not find unused die with value ${dieValue}`);
      return;
    }
    
    // Mark this die as used
    this.usedDice[dieIndex] = true;
    this.diceEls[dieIndex].classList.remove('selected');
    this.diceEls[dieIndex].classList.add('used');
  }
  
  // Mark a die as available again (e.g., after undo)
  markDieAvailable(dieValue) {
    let dieIndex = -1;
    
    // Find which die matches this value
    for (let i = 0; i < this.dice.length; i++) {
      if (this.dice[i] === dieValue && this.usedDice[i]) {
        dieIndex = i;
        break;
      }
    }
    
    if (dieIndex === -1) {
      console.warn(`Could not find used die with value ${dieValue}`);
      return;
    }
    
    // Mark this die as available
    this.usedDice[dieIndex] = false;
    this.diceEls[dieIndex].classList.remove('used');
  }
  
  // Calculate and return the move distance based on from and to positions
  calculateMoveDistance(fromPos, toPos) {
    // For bearing off, special calculation
    if (toPos === -1 || toPos === 'off') {
      return fromPos + 1; // Distance to bear off is position + 1
    }
    
    // Regular move
    return Math.abs(fromPos - toPos);
  }
  
  // Show a message about dice state
  showMessage(message) {
    this.messageEl.textContent = message;
    this.messageEl.style.display = 'block';
  }
  
  // Event handlers
  handleFirstMoveComplete(event) {
    // First move has been made, need to mark one die as used
    if (event.detail && event.detail.usedDie) {
      this.markDieUsed(event.detail.usedDie);
    }
  }
  
  handleBoardUpdated(event) {
    // Reset dice state if it's a new turn
    if (event.detail && event.detail.newTurn) {
      this.updateDice(event.detail.dice || this.dice);
    }
  }
  
  handleDiceRolled(event) {
    if (event.detail && event.detail.dice) {
      this.updateDice(event.detail.dice);
      
      // Check if this is the first turn special case
      if (event.detail.firstTurn && event.detail.isSpecialDoubles) {
        this.isFirstTurn = true;
        
        // Highlight that this is special doubles with a message
        this.diceEls.forEach(dieEl => dieEl.classList.add('special-doubles'));
        
        // Add a message about the special rule
        this.showMessage('Special case: You can move up to 2 checkers from the head position');
        
        // Dispatch an event to show the special doubles notification
        const notifyEvent = new CustomEvent('specialDoublesDetected', {
          bubbles: true,
          detail: { dice: event.detail.dice }
        });
        window.dispatchEvent(notifyEvent);
      } else {
        this.isFirstTurn = event.detail.firstTurn || false;
      }
    }
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = DiceComponent;
}

window.DiceComponent = DiceComponent;
