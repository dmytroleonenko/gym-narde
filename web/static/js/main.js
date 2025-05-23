/**
 * Main entry point for the application
 */

// Initialize on document ready
document.addEventListener('DOMContentLoaded', function() {
  console.log('Initializing Long Nardy application');
  
  // Ensure container element with id "backgammon" exists
  if (!document.getElementById('backgammon')) {
    const container = document.createElement('div');
    container.id = 'backgammon';
    document.body.appendChild(container);
  }
  
  // Create and initialize app with container ID
  const app = new App('backgammon');
  window.api = app.api;
  console.log("window.api has been set:", window.api);
  // Store app in global scope for debugging
  window.app = app;
  
  console.log('Long Nardy application initialized');
  
  // Show ready message
  const statusEl = document.querySelector('.action-status');
  if (statusEl) {
    statusEl.textContent = 'Game ready - Press "New Game" to start';
  }
});
