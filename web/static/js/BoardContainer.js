class BoardContainer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.init();
  }

  init() {
    // Create main board element
    this.boardEl = document.createElement('div');
    this.boardEl.id = 'board-container';
    this.boardEl.className = 'board';
    this.container.appendChild(this.boardEl);

    // Create board rows for top and bottom points
    this.topRow = document.createElement('div');
    this.topRow.className = 'row top';
    this.boardEl.appendChild(this.topRow);

    this.bottomRow = document.createElement('div');
    this.bottomRow.className = 'row bottom';
    this.boardEl.appendChild(this.bottomRow);

    // Add notifications for rule visualizations
    this.addRuleNotifications();

    // Initialize sub-components
    this.initPoints();
    this.initBearingOff();
    this.initActionPanel();
    this.initDicePanel();
    this.bindEvents();
  }

  addRuleNotifications() {
    // Add head rule notification
    this.headRuleNotification = document.createElement('div');
    this.headRuleNotification.className = 'head-rule-notification';
    this.headRuleNotification.textContent = 'Only 1 checker may leave the head position per turn';
    this.boardEl.appendChild(this.headRuleNotification);

    // Add special doubles notification
    this.specialDoublesNotification = document.createElement('div');
    this.specialDoublesNotification.className = 'special-doubles-notification';
    this.specialDoublesNotification.textContent = 'First turn with doubles 3, 4, or 6: You can move 2 checkers from the head';
    this.boardEl.appendChild(this.specialDoublesNotification);

    // Add block rule warning
    this.blockRuleWarning = document.createElement('div');
    this.blockRuleWarning.className = 'block-rule-warning';
    this.blockRuleWarning.style.display = 'none';
    this.blockRuleWarning.textContent = 'Warning: This move would create a block of 6 checkers with no opponent pieces ahead, which is not allowed';
    this.boardEl.appendChild(this.blockRuleWarning);

    // Add block rule indicator container
    this.blockRuleIndicators = document.createElement('div');
    this.blockRuleIndicators.className = 'block-rule-indicators';
    this.boardEl.appendChild(this.blockRuleIndicators);
  }

  initPoints() {
    // Create point components for all 24 positions in Long Nardy
    this.points = [];
    
    // Points are numbered 0-23
    // 0-11 in top row (left to right)
    // 12-23 in bottom row (right to left) - reverse order for proper display
    
    // Create top row points (0-11)
    for (let i = 0; i < 12; i++) {
      const point = document.createElement('div');
      point.className = 'point';
      point.id = `point-${i}`;
      point.dataset.index = i;
      
      // Point 11 is black's head position
      if (i === 11) {
        point.classList.add('head-position');
      }
      
      // Points 0-5 are white's home area (bearing off zone)
      if (i >= 0 && i <= 5) {
        point.classList.add('home-area');
      }
      
      this.topRow.appendChild(point);
      this.points[i] = point;
    }
    
    // Create bottom row points (12-23) - in reversed visual order (right to left)
    for (let i = 23; i >= 12; i--) {
      const point = document.createElement('div');
      point.className = 'point';
      point.id = `point-${i}`;
      point.dataset.index = i;
      
      // Point 23 is white's head position
      if (i === 23) {
        point.classList.add('head-position');
      }
      
      // Points 12-17 are black's home area (bearing off zone)
      if (i >= 12 && i <= 17) {
        point.classList.add('home-area');
      }
      
      this.bottomRow.appendChild(point);
      this.points[i] = point;
    }
  }

  initBearingOff() {
    // Create bearing off area
    this.bearingOff = document.createElement('div');
    this.bearingOff.id = 'bearing-off';
    this.bearingOff.className = 'bearing-off';
    this.boardEl.appendChild(this.bearingOff);
  }

  initActionPanel() {
    // Create action panel component
    this.actionPanel = document.createElement('div');
    this.actionPanel.id = 'action-panel';
    this.actionPanel.className = 'action-panel';
    
    // Roll, Confirm, Undo buttons
    this.rollBtn = document.createElement('button');
    this.rollBtn.className = 'action-button primary';
    this.rollBtn.id = 'btn-roll';
    this.rollBtn.textContent = 'Roll';
    
    this.confirmBtn = document.createElement('button');
    this.confirmBtn.className = 'action-button primary';
    this.confirmBtn.id = 'btn-confirm';
    this.confirmBtn.textContent = 'Confirm';
    
    this.undoBtn = document.createElement('button');
    this.undoBtn.className = 'action-button secondary';
    this.undoBtn.id = 'btn-undo';
    this.undoBtn.textContent = 'Undo';
    
    this.actionPanel.appendChild(this.rollBtn);
    this.actionPanel.appendChild(this.confirmBtn);
    this.actionPanel.appendChild(this.undoBtn);
    
    // Add status message area
    this.statusMessage = document.createElement('div');
    this.statusMessage.className = 'action-status';
    this.actionPanel.appendChild(this.statusMessage);
    
    this.container.appendChild(this.actionPanel);
  }

  initDicePanel() {
    // Create dice panel component
    this.dicePanel = document.createElement('div');
    this.dicePanel.id = 'dice-panel';
    this.dicePanel.className = 'dice-container';
    this.container.appendChild(this.dicePanel);
  }

  bindEvents() {
    // Bind events to buttons and dispatch custom events for the Comm layer
    this.rollBtn.addEventListener('click', () => {
      this.dispatchActionEvent('roll');
    });
    
    this.confirmBtn.addEventListener('click', () => {
      this.dispatchActionEvent('confirm');
    });
    
    this.undoBtn.addEventListener('click', () => {
      this.dispatchActionEvent('undo');
    });
    
    // Bind events to point clicks
    this.points.forEach(point => {
      point.addEventListener('click', (e) => {
        const index = parseInt(e.currentTarget.dataset.index, 10);
        this.dispatchPointEvent(index);
      });
    });
    
    // Bind event to bearing off area
    this.bearingOff.addEventListener('click', () => {
      this.dispatchBearingOffEvent();
    });
  }

  dispatchActionEvent(action) {
    const event = new CustomEvent('actionTriggered', {
      bubbles: true,
      detail: { action }
    });
    window.dispatchEvent(event);
  }

  dispatchPointEvent(index) {
    const event = new CustomEvent('pointSelected', {
      bubbles: true,
      detail: { index }
    });
    window.dispatchEvent(event);
  }

  dispatchBearingOffEvent() {
    const event = new CustomEvent('bearingOffSelected', {
      bubbles: true,
      detail: {}
    });
    window.dispatchEvent(event);
  }

  // Long Nardy rule visualization methods
  
  showHeadRuleNotification(active = true) {
    if (active) {
      // First remove active to reset animation
      this.headRuleNotification.classList.remove('active');
      
      // Force a browser reflow to restart the animation
      void this.headRuleNotification.offsetWidth;
      
      // Now add active to trigger animation from start
      this.headRuleNotification.classList.add('active');
      
      // Auto-hide after 3 seconds
      setTimeout(() => {
        this.headRuleNotification.classList.remove('active');
      }, 3000);
    } else {
      this.headRuleNotification.classList.remove('active');
    }
  }

  showSpecialDoublesNotification(active = true) {
    if (active) {
      // First remove active to reset animation
      this.specialDoublesNotification.classList.remove('active');
      
      // Force a browser reflow to restart the animation
      void this.specialDoublesNotification.offsetWidth;
      
      // Now add active to trigger animation from start
      this.specialDoublesNotification.classList.add('active');
      
      // Auto-hide after 5 seconds
      setTimeout(() => {
        this.specialDoublesNotification.classList.remove('active');
      }, 5000);
    } else {
      this.specialDoublesNotification.classList.remove('active');
    }
  }

  showBlockRuleWarning(active = true) {
    if (active) {
      // Create a new warning element instead of reusing the same one
      // This ensures the animation plays every time
      this.blockRuleWarning.style.display = 'none';
      
      const newWarning = this.blockRuleWarning.cloneNode(true);
      newWarning.style.display = 'block';
      
      // Remove the old warning if it exists
      if (this.blockRuleWarning.parentNode) {
        this.blockRuleWarning.parentNode.replaceChild(newWarning, this.blockRuleWarning);
      }
      
      this.blockRuleWarning = newWarning;
      
      // The animation will handle hiding automatically due to fadeInOut
    } else {
      this.blockRuleWarning.style.display = 'none';
    }
  }

  // Update this method to visualize contiguous blocks
  updateBlockRuleIndicators(blocks) {
    // Clear existing indicators
    this.blockRuleIndicators.innerHTML = '';
    
    if (!blocks || blocks.length === 0) return;
    
    blocks.forEach(block => {
      const { start, end, width } = block;
      
      const indicator = document.createElement('div');
      indicator.className = 'indicator';
      
      // Position the indicator based on the block position
      // This might need adjustments based on the board layout
      const startPercent = (start / 24) * 100;
      const widthPercent = (width / 24) * 100;
      
      indicator.style.left = `${startPercent}%`;
      indicator.style.width = `${widthPercent}%`;
      
      this.blockRuleIndicators.appendChild(indicator);
    });
  }
  
  // Add this method to highlight the home areas
  highlightHomeAreas(player) {
    // Reset all home area highlights
    document.querySelectorAll('.home-area').forEach(element => {
      element.style.borderColor = '#4CAF50';
    });
    
    if (player === 'white') {
      // Highlight white's home (points 0-5)
      for (let i = 0; i <= 5; i++) {
        const point = document.getElementById(`point-${i}`);
        if (point) point.style.borderColor = '#2196F3';
      }
    } else if (player === 'black') {
      // Highlight black's home (points 12-17)
      for (let i = 12; i <= 17; i++) {
        const point = document.getElementById(`point-${i}`);
        if (point) point.style.borderColor = '#2196F3';
      }
    }
  }
  
  // Add method to visualize AI moves
  visualizeAIMoves(aiMoves, isWinningMove = false) {
    if (!aiMoves || aiMoves.length === 0) return;
    
    // Clear any existing visualizations
    document.querySelectorAll('.ai-move-indicator').forEach(el => el.remove());
    
    // Create temporary elements to visualize each move
    aiMoves.forEach((move, index) => {
      // Get positions
      const fromPos = move.from;
      const toPos = move.to;
      
      // Create the indicator element
      const indicator = document.createElement('div');
      indicator.className = 'ai-move-indicator';
      indicator.dataset.moveIndex = index;
      
      // Set a different color for the final winning move
      if (isWinningMove && index === aiMoves.length - 1) {
        indicator.classList.add('winning-move');
      }
      
      // Get fromPoint element position
      const fromPoint = document.getElementById(`point-${fromPos}`);
      if (!fromPoint) return;
      
      const fromRect = fromPoint.getBoundingClientRect();
      const boardRect = this.boardEl.getBoundingClientRect();
      
      // Position the start point relative to the board
      const startX = fromRect.left - boardRect.left + fromRect.width / 2;
      const startY = fromRect.top - boardRect.top + fromRect.height / 2;
      
      // Set end point based on toPos (could be bearing off)
      let endX, endY;
      
      if (toPos === -1) {
        // Bearing off
        const bearingOff = document.getElementById('bearing-off');
        if (bearingOff) {
          const bearingOffRect = bearingOff.getBoundingClientRect();
          endX = bearingOffRect.left - boardRect.left + bearingOffRect.width / 2;
          endY = bearingOffRect.top - boardRect.top + bearingOffRect.height / 2;
        }
      } else {
        // Moving to another point
        const toPoint = document.getElementById(`point-${toPos}`);
        if (toPoint) {
          const toRect = toPoint.getBoundingClientRect();
          endX = toRect.left - boardRect.left + toRect.width / 2;
          endY = toRect.top - boardRect.top + toRect.height / 2;
        }
      }
      
      if (endX === undefined || endY === undefined) return;
      
      // Now create an SVG arrow to indicate the move
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('width', '100%');
      svg.setAttribute('height', '100%');
      svg.style.position = 'absolute';
      svg.style.top = '0';
      svg.style.left = '0';
      svg.style.pointerEvents = 'none';
      svg.style.zIndex = '50';
      
      const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      
      // Calculate arrow path 
      const dx = endX - startX;
      const dy = endY - startY;
      const angle = Math.atan2(dy, dx);
      
      // Make arrow shorter to not overlap with pieces
      const headlen = 10; // arrow head length
      const shortening = 15; // shorten both ends by this amount
      
      // Shorten both ends
      const shortenedStartX = startX + shortening * Math.cos(angle);
      const shortenedStartY = startY + shortening * Math.sin(angle);
      const shortenedEndX = endX - shortening * Math.cos(angle);
      const shortenedEndY = endY - shortening * Math.sin(angle);
      
      // Arrow head coordinates
      const arrowHead1X = shortenedEndX - headlen * Math.cos(angle - Math.PI / 6);
      const arrowHead1Y = shortenedEndY - headlen * Math.sin(angle - Math.PI / 6);
      const arrowHead2X = shortenedEndX - headlen * Math.cos(angle + Math.PI / 6);
      const arrowHead2Y = shortenedEndY - headlen * Math.sin(angle + Math.PI / 6);
      
      // Create path 
      const pathData = `M ${shortenedStartX} ${shortenedStartY} 
                        L ${shortenedEndX} ${shortenedEndY}
                        M ${shortenedEndX} ${shortenedEndY} 
                        L ${arrowHead1X} ${arrowHead1Y}
                        M ${shortenedEndX} ${shortenedEndY} 
                        L ${arrowHead2X} ${arrowHead2Y}`;
      
      arrow.setAttribute('d', pathData);
      arrow.setAttribute('stroke', isWinningMove && index === aiMoves.length - 1 ? '#f44336' : '#2196F3');
      arrow.setAttribute('stroke-width', '3');
      arrow.setAttribute('fill', 'none');
      arrow.style.opacity = '0.7';
      
      // Add animation
      const animate = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
      animate.setAttribute('attributeName', 'opacity');
      animate.setAttribute('values', '0.7;0.3;0.7');
      animate.setAttribute('dur', '2s');
      animate.setAttribute('repeatCount', 'indefinite');
      arrow.appendChild(animate);
      
      // Add move number
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', (shortenedStartX + shortenedEndX) / 2);
      text.setAttribute('y', (shortenedStartY + shortenedEndY) / 2 - 5);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', isWinningMove && index === aiMoves.length - 1 ? '#f44336' : '#2196F3');
      text.setAttribute('font-size', '12px');
      text.setAttribute('font-weight', 'bold');
      text.textContent = `${index + 1}`;
      
      svg.appendChild(arrow);
      svg.appendChild(text);
      indicator.appendChild(svg);
      
      this.boardEl.appendChild(indicator);
      
      // Auto-remove after 5 seconds (longer for winning moves)
      setTimeout(() => {
        indicator.remove();
      }, isWinningMove ? 7000 : 5000);
    });
  }
}

// Export module if using module system
if (typeof module !== 'undefined' && module.exports) {
  module.exports = BoardContainer;
}

// If in browser, attach to window
window.BoardContainer = BoardContainer;
