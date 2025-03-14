class PieceComponent {
  constructor(piece, container) {
    this.piece = piece;
    this.container = container;
    this.isDragging = false;
    this.initialPosition = { x: 0, y: 0 };
    this.dragOffset = { x: 0, y: 0 };
    this.init();
  }

  init() {
    this.el = document.createElement('div');
    this.el.className = `piece-component ${this.piece.color}`;
    this.el.dataset.id = this.piece.id;
    this.el.dataset.position = this.piece.position;
    this.el.textContent = this.piece.label || '';
    this.bindEvents();
    this.container.appendChild(this.el);
  }

  bindEvents() {
    // Mouse events
    this.el.addEventListener('mousedown', this.handleDragStart.bind(this));
    document.addEventListener('mousemove', this.handleDragMove.bind(this));
    document.addEventListener('mouseup', this.handleDragEnd.bind(this));

    // Touch events for mobile
    this.el.addEventListener('touchstart', this.handleDragStart.bind(this), { passive: false });
    document.addEventListener('touchmove', this.handleDragMove.bind(this), { passive: false });
    document.addEventListener('touchend', this.handleDragEnd.bind(this));
    
    // Click event as fallback
    this.el.addEventListener('click', (e) => {
      if (!this.hasMoved) {
        this.onClick();
      }
    });
  }

  handleDragStart(e) {
    // Prevent default to avoid selection, etc.
    e.preventDefault();
    
    // Don't start drag if this piece can't move
    if (this.piece.color !== 'white' || !this.piece.canMove) {
      // Add a subtle animation to indicate this piece can't be moved
      if (this.piece.color === 'white' && !this.piece.canMove) {
        this.el.style.transition = 'transform 0.15s ease';
        this.el.style.transform = 'scale(1.05)';
        
        setTimeout(() => {
          this.el.style.transform = 'scale(1)';
          
          setTimeout(() => {
            this.el.style.transition = '';
          }, 150);
        }, 150);
      }
      return;
    }
    
    this.isDragging = true;
    this.hasMoved = false;
    
    // Make sure there's no transition active
    this.el.style.transition = '';
    
    // Get current element position
    const rect = this.el.getBoundingClientRect();
    this.initialPosition = {
      x: rect.left,
      y: rect.top
    };

    // Calculate offset within the piece where the drag started
    const clientX = e.clientX || e.touches[0].clientX;
    const clientY = e.clientY || e.touches[0].clientY;
    
    this.dragOffset = {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
    
    // Add dragging class with visual feedback
    this.el.classList.add('dragging');
    
    // Add a subtle scale effect when starting the drag
    this.el.style.transform = 'scale(1.1)';
    
    // Dispatch pieceSelected event to get valid moves
    const event = new CustomEvent('pieceSelected', { 
      detail: { 
        piece: this.piece,
        dragStarted: true
      } 
    });
    window.dispatchEvent(event);
    
    // Add a slight delay before starting the drag motion
    // This gives time for the scale effect to be visible
    setTimeout(() => {
      if (this.isDragging) {
        // Initialize position
        this.updatePosition(clientX, clientY);
      }
    }, 50);
  }

  handleDragMove(e) {
    if (!this.isDragging) return;
    
    e.preventDefault();
    
    // Get cursor position
    const clientX = e.clientX || (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
    const clientY = e.clientY || (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
    
    // Update position
    this.updatePosition(clientX, clientY);
    this.hasMoved = true;
    
    // Dispatch piece moving event
    const event = new CustomEvent('pieceMoving', { 
      detail: { 
        piece: this.piece,
        x: clientX,
        y: clientY
      } 
    });
    window.dispatchEvent(event);
  }

  handleDragEnd(e) {
    if (!this.isDragging) return;
    
    this.isDragging = false;
    this.el.classList.remove('dragging');
    
    // Reset styles
    this.el.style.boxShadow = '';
    this.el.style.zIndex = '';
    
    // Get cursor position
    const clientX = e.clientX || (e.changedTouches && e.changedTouches[0] ? e.changedTouches[0].clientX : 0);
    const clientY = e.clientY || (e.changedTouches && e.changedTouches[0] ? e.changedTouches[0].clientY : 0);
    
    // Dispatch piece dropped event
    const event = new CustomEvent('pieceDropped', { 
      detail: { 
        piece: this.piece,
        x: clientX,
        y: clientY
      } 
    });
    window.dispatchEvent(event);
    
    // Reset position with a smooth transition back
    // This will be overridden if the drop is valid when board rerenders
    setTimeout(() => {
      this.el.style.transition = 'transform 0.3s ease-in-out';
      this.el.style.transform = 'translate3d(0, 0, 0)';
      
      // Remove transition after animation completes
      setTimeout(() => {
        this.el.style.transition = '';
      }, 300);
    }, 50);
  }

  updatePosition(clientX, clientY) {
    const left = clientX - this.dragOffset.x;
    const top = clientY - this.dragOffset.y;
    
    // Use transform for better performance
    this.el.style.transform = `translate3d(${left - this.initialPosition.x}px, ${top - this.initialPosition.y}px, 0)`;
    
    // Add scale effect during drag
    if (this.isDragging) {
      this.el.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.3)';
      this.el.style.zIndex = '100';
    }
  }

  onClick() {
    console.log(`Piece ${this.piece.id} clicked`);
    const event = new CustomEvent('pieceSelected', { detail: { piece: this.piece } });
    window.dispatchEvent(event);
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PieceComponent;
}

window.PieceComponent = PieceComponent;
