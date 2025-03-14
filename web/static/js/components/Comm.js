/**
 * Communications module for handling events
 * 
 * @constructor
 */
function Comm() {
  this.subscribers = {};
  this.init();
}

// Message types
Comm.Message = {
  // Game events
  EVENT_MATCH_START: 'match_start',
  EVENT_MATCH_OVER: 'match_over',
  EVENT_TURN_START: 'turn_start',
  EVENT_DICE_ROLLED: 'dice_rolled',
  EVENT_MOVE_EXECUTED: 'move_executed',
  EVENT_MOVES_CONFIRMED: 'moves_confirmed',
  EVENT_MOVES_UNDONE: 'moves_undone',
  EVENT_PLAYER_JOINED: 'player_joined',
  
  // Requests
  JOIN_MATCH: 'join_match',
  CREATE_MATCH: 'create_match',
  CREATE_GUEST: 'create_guest',
  REQ_ROLL_DICE: 'roll_dice',
  REQ_MOVE: 'move',
  REQ_UP: 'up',
  REQ_CONFIRM: 'confirm',
  REQ_UNDO: 'undo',
  REQ_RESIGN: 'resign',
  REQ_PLAY_RANDOM: 'play_random'
};

Comm.prototype = {
  /**
   * Initialize communications
   */
  init: function() {
    console.log('Communication system initialized');
  },
  
  /**
   * Subscribe to an event
   * 
   * @param {string} eventName - Name of event to subscribe to
   * @param {function} callback - Callback function when event occurs
   */
  subscribe: function(eventName, callback) {
    if (!this.subscribers[eventName]) {
      this.subscribers[eventName] = [];
    }
    this.subscribers[eventName].push(callback);
    console.log('Subscribed to event:', eventName);
  },
  
  /**
   * Dispatch an event to all subscribers
   * 
   * @param {string} eventName - Name of event to dispatch
   * @param {Object} payload - Event data payload
   */
  dispatch: function(eventName, payload) {
    console.log('Dispatching event:', eventName, payload);
    if (this.subscribers[eventName]) {
      this.subscribers[eventName].forEach(function(callback) {
        callback(payload);
      });
    }
  },
  
  /**
   * Send a message to the server
   * 
   * @param {string} msg - Message type
   * @param {Object} payload - Message data payload
   * @param {function} [callback] - Optional callback for response
   */
  send: function(msg, payload, callback) {
    console.log('Sending message:', msg, payload);
    switch (msg) {
      case 'getValidMoves':
        return window.api.reqGetValidMoves(payload.position);
      case 'makeMove':
        return window.api.reqMove(null, payload.fromPosition, payload.toPosition);
      case 'rollDice':
      case 'roll':
        return window.api.reqRollDice();
      case 'confirmMove':
      case 'confirm':
        return window.api.reqConfirmMoves();
      case 'undoMove':
      case 'undo':
        return window.api.reqUndoMoves();
      case 'newGame':
        return window.api.reqNewGame();
      default:
        console.error('Unknown message type in send:', msg);
        return;
    }
  }
};

// Create a singleton instance
var comm = new Comm();

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = comm;
}
