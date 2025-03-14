/**
 * Main entry point for the application
 */

// Initialize on document ready
document.addEventListener('DOMContentLoaded', function() {
  console.log('Initializing Long Nardy application');
  
  // Create and initialize app with container ID
  const app = new App('backgammon');
  // Make sure the Comm module can find the API wrapper
  window.api = app.api;
  
  // Start a new game automatically after App is fully initialized, ensuring window.api is available
  function trySendNewGame() {
    if (window.api && window.api.reqNewGame) {
      comm.send('newGame');
    } else {
      console.log('Waiting for API to be available...');
      setTimeout(trySendNewGame, 500);
    }
  }
  trySendNewGame();
  
  // Store app in global scope for debugging
  window.app = app;
  
  console.log('Long Nardy application initialized');
  
  // Show ready message
  const statusEl = document.querySelector('.action-status');
  if (statusEl) {
    statusEl.textContent = 'Game ready - Press "New Game" to start';
  }
});
