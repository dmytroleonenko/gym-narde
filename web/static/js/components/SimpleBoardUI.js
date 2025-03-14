/**
 * Contains graphical user interface and functionality for moving pieces
 *
 * @constructor
 * @param {Client} client - Client object in control of this UI
 */
function SimpleBoardUI(client) {
  /**
   * Client object
   * @type {Client}
   */
  this.client = client;

  /**
   * Current match
   * @type {Match}
   */
  this.match = null;

  /**
   * Game rule
   * @type {Rule}
   */
  this.rule = null;

  /**
   * Fields of the board
   * @type {Array}
   */
  this.fields = [];

  /**
   * Random rotation angles for dice
   * @type {Array}
   */
  this.rotationAngle = [0, 0];

  // Initialize the UI
  this.init();
}

SimpleBoardUI.prototype = {
  /**
   * Initialize the UI
   */
  init: function() {
    // Get container from client config
    this.container = $('#' + this.client.config.containerID);

    // Add board template
    this.container.append($('#tmpl-board').html());
    this.container.append($('<div id="ohsnap"></div>'));

    // Store board element
    this.board = $('.board');

    // Store field elements
    this.fields = [];
    for (var i = 0; i < 4; i++) {
      this.fields[i] = $('#field' + i);
    }

    // Create board points
    this.createPoints();

    // Set up event handlers
    this.assignActions();

    // Randomize dice rotation angles
    this.randomizeDiceRotation();

    console.log('SimpleBoardUI initialized, points created');
  },

  /**
   * Rounds down floating point value to specified number of digits
   * after decimal point
   *
   * @param {Number} number - float number to round
   * @param {Number} digits - number of digits after decimal point
   * @returns {Number} rounded number as float
   */
  toFixedDown: function(number, digits) {
    var n = number - Math.pow(10, -digits)/2;
    n += n / Math.pow(2, 53); // Added for precision
    return n.toFixed(digits);
  },

  /**
   * Show a notification message
   *
   * @param {string} message - Message to display
   * @param {Object} params - Optional parameters
   */
  notifyOhSnap: function(message, params) {
    if (window.ohSnap) {
      window.ohSnap(message, params);
    } else {
      console.log('Notification:', message);
    }
  },

  /**
   * Show an information notification
   *
   * @param {string} message - Message to display
   * @param {number} [timeout=5000] - Time in ms before notification disappears
   */
  notifyInfo: function(message, timeout) {
    this.notifyOhSnap(message, {color: 'blue', duration: timeout});
  },

  /**
   * Show a positive notification
   *
   * @param {string} message - Message to display
   * @param {number} [timeout=5000] - Time in ms before notification disappears
   */
  notifyPositive: function(message, timeout) {
    this.notifyOhSnap(message, {color: 'green', duration: timeout});
  },

  /**
   * Show a negative notification
   *
   * @param {string} message - Message to display
   * @param {number} [timeout=5000] - Time in ms before notification disappears
   */
  notifyNegative: function(message, timeout) {
    this.notifyOhSnap(message, {color: 'red', duration: timeout});
  },

  /**
   * Get DOM element for a point at a specific position
   *
   * @param {number} pos - Point position
   * @returns {jQuery} Point element
   */
  getPointElem: function(pos) {
    return $('#point' + pos);
  },

  /**
   * Get DOM element for a piece
   *
   * @param {Piece} piece - Piece object
   * @returns {jQuery} Piece element
   */
  getPieceElem: function(piece) {
    return $('#piece' + piece.id);
  },

  /**
   * Get the top piece element at a position
   *
   * @param {number} pos - Point position
   * @returns {jQuery} Top piece element
   */
  getTopPieceElem: function(pos) {
    var pointElem = $('#point' + pos);
    return pointElem.find('div.piece').last();
  },

  /**
   * Get the top piece object at a position
   *
   * @param {number} pos - Point position
   * @returns {Piece} Top piece object
   */
  getTopPiece: function(pos) {
    var pieceElem = this.getTopPieceElem(pos);
    return pieceElem.data('piece');
  },

  /**
   * Get DOM element for the bar of a specific piece type
   *
   * @param {PieceType} type - Piece type (white or black)
   * @returns {jQuery} Bar element
   */
  getBarElem: function(type) {
    var barID = type === model.PieceType.WHITE ? 'bottom-bar' : 'top-bar';
    return $('#' + barID);
  },

  /**
   * Handles clicking on a point (position)
   * Add [SHIFT] to move UP
   *
   * @param {Event} e - Click event
   */
  handlePointClick: function(e) {
    e.preventDefault();
    var self = e.data;

    // Ensure we have a match and game
    if (!self.match || !self.match.currentGame) {
      console.log('No active game');
      return;
    }

    var game = self.match.currentGame;
    var position = $(e.currentTarget).data('position');

    console.log('Point clicked:', position);

    // Determine if this is the first click (piece selection) or second click (destination)
    if (self.selectedPosition === undefined) {
      // First click - piece selection
      var piece = self.getTopPiece(position);

      // Check if there's a piece at this position
      if (!piece) {
        console.log('No piece at position', position);
        return;
      }

      // Check if it's the player's piece
      if (piece.type !== self.client.player.currentPieceType) {
        console.log('Not your piece');
        return;
      }

      // Check if the game has moves left
      if (!game.turnDice || !game.turnDice.movesLeft || game.turnDice.movesLeft.length === 0) {
        console.log('No moves left');
        return;
      }

      // This is a valid piece selection
      self.selectedPosition = position;
      console.log('Piece selected at position', position);

      // Highlight selected piece
      $(e.currentTarget).addClass('selected');

      // Get valid destinations for this piece
      self.getValidDestinations(position);
    } else {
      // Second click - destination selection

      // Check if clicking on the same position (cancel selection)
      if (position === self.selectedPosition) {
        console.log('Selection canceled');
        self.clearSelection();
        return;
      }

      // Check if this is a valid destination
      if (self.validDestinations.indexOf(position) === -1) {
        console.log('Invalid destination');
        return;
      }

      // Valid destination - make the move
      console.log('Moving from', self.selectedPosition, 'to', position);

      // Get the piece at selected position
      var piece = self.getTopPiece(self.selectedPosition);

      // Get the die value for this move
      var steps = Math.abs(self.selectedPosition - position);

      // Execute the move
      self.client.reqMove(piece, self.selectedPosition, position);

      // Clear selection state
      self.clearSelection();
    }
  },

  /**
   * Get valid destinations for a position
   *
   * @param {number} position - Position to get valid destinations for
   */
  getValidDestinations: function(position) {
    var game = this.match.currentGame;

    // Check if there are valid moves in the game state
    if (!game.validMoves || !game.validMoves[position]) {
      console.log('No valid moves for position', position);
      this.validDestinations = [];
      return;
    }

    // Extract valid destinations from moves
    this.validDestinations = [];
    var movesForPosition = game.validMoves[position];

    for (var i = 0; i < movesForPosition.length; i++) {
      var move = movesForPosition[i];
      var destination = move[1]; // The second element is the destination

      if (destination === 'off') {
        destination = -1; // Use -1 to represent bearing off
      }

      this.validDestinations.push(destination);
    }

    console.log('Valid destinations for position', position, ':', this.validDestinations);

    // Highlight valid destinations
    this.highlightValidDestinations();
  },

  /**
   * Highlight valid destinations on the board
   */
  highlightValidDestinations: function() {
    // Clear existing highlights
    $('.point').removeClass('valid-destination');
    $('#bearing-off').removeClass('valid-destination');

    // Highlight valid points
    for (var i = 0; i < this.validDestinations.length; i++) {
      var destination = this.validDestinations[i];

      if (destination === -1) {
        // Highlight bearing off area
        $('#bearing-off').addClass('valid-destination');
      } else {
        // Highlight point
        $('#point' + destination).addClass('valid-destination');
      }
    }
  },

  /**
   * Clear selection state
   */
  clearSelection: function() {
    // Clear selection variables
    this.selectedPosition = undefined;
    this.validDestinations = [];

    // Clear UI highlights
    $('.point').removeClass('selected valid-destination');
    $('#bearing-off').removeClass('valid-destination');
  },

  /**
   * Handles clicking on bar
   *
   * @param {Event} e - Click event
   */
  handleBarClick: function(e) {
    var self = e.data;
    var game = self.match.currentGame;

    // If no more moves, do nothing
    if (!model.Game.hasMoreMoves(game)) {
      return;
    }

    var pieceElem = $(e.currentTarget).find('div.piece').last();
    var piece = pieceElem.data('piece');

    // If no piece on bar, do nothing
    if (!piece) {
      return;
    }

    // If not this player's piece, do nothing
    if (piece.type !== self.client.player.currentPieceType) {
      return;
    }

    // Get next die value
    var steps = game.turnDice.movesLeft[0];

    // Move from bar by steps
    self.client.reqRecover(piece, steps);
  },

  /**
   * Assign actions to DOM elements
   */
  assignActions: function() {
    var self = this;

    console.log('Assigning action handlers');

    // Game actions - Roll dice button
    $('#btn-roll').unbind('click');
    $('#btn-roll').click(function() {
      console.log('Roll button clicked');
      self.client.reqRollDice();
    });

    // Confirm moves button
    $('#btn-confirm').unbind('click');
    $('#btn-confirm').click(function() {
      console.log('Confirm button clicked');
      self.compactAllPositions();
      self.client.reqConfirmMoves();
    });

    // Undo moves button
    $('#btn-undo').unbind('click');
    $('#btn-undo').click(function() {
      console.log('Undo button clicked');
      self.client.reqUndoMoves();
    });

    // Menu undo button
    $('#menu-undo').unbind('click');
    $('#menu-undo').click(function() {
      console.log('Menu undo clicked');
      $('.navbar').collapse('hide');
      self.client.reqUndoMoves();
    });

    // Menu resign button
    $('#menu-resign').unbind('click');
    $('#menu-resign').click(function() {
      console.log('Menu resign clicked');

      // Ask for confirmation
      if (confirm('Are you sure you want to resign the game?')) {
        $('.navbar').collapse('hide');
        self.client.reqResignGame();
      }
    });

    // Menu close button
    $('#menu-close').unbind('click');
    $('#menu-close').click(function() {
      console.log('Menu close clicked');
      $('.navbar').collapse('hide');
    });

    // Collapsible navbar toggle
    $('.navbar-toggle').unbind('click');
    $('.navbar-toggle').click(function() {
      console.log('Navbar toggle clicked');
      $('.navbar').collapse('toggle');
    });

    // Points will be assigned actions when created
    console.log('Action handlers assigned');
  },

  /**
   * Create a point (triangle) on the board
   *
   * @param {jQuery} field - Field element to add point to
   * @param {number} pos - Position of the point
   * @param {string} typeClass - CSS class for point type (odd/even)
   */
  createPoint: function(field, pos, typeClass) {
    var pointElem = $('<div id="point' + pos + '" class="point ' + typeClass + '"></div>');
    pointElem.data('position', pos);
    field.append(pointElem);

    // Add head and home area indicators for Long Nardy
    if (pos === 23) { // White head
      pointElem.addClass('head-position');
      console.log('Added head-position class to point 23 (White head)');
    } else if (pos === 11) { // Black head
      pointElem.addClass('head-position');
      console.log('Added head-position class to point 11 (Black head)');
    } else if (pos >= 0 && pos <= 5) { // White home
      pointElem.addClass('home-area');
      console.log('Added home-area class to point', pos, '(White home)');
    } else if (pos >= 12 && pos <= 17) { // Black home
      pointElem.addClass('home-area');
      console.log('Added home-area class to point', pos, '(Black home)');
    }

    // Assign click handler
    var self = this;
    pointElem.unbind('mousedown');
    pointElem.mousedown(self, self.handlePointClick);

    console.log('Created point', pos, 'with click handler');
  },

  /**
   * Create all points on the board
   */
  createPoints: function() {
    // Fields are arranged on the board like this:
    // +12-13-14-15-16-17------18-19-20-21-22-23-+
    // |                  |   |                  |
    // |      Field 0     |   |      Field 2     |
    // |                  |   |                  |
    // |                  |   |                  |
    // |                  |   |                  |
    // |                  |   |                  |
    // |                  |   |                  |
    // |                  |   |                  |
    // |                  |   |                  |
    // |      Field 1     |   |      Field 3     |
    // |                  |   |                  |
    // +11-10--9--8--7--6-------5--4--3--2--1--0-+ -1

    // Top left field (12-17)
    for (var k = 12; k <= 17; k++) {
      var typeClass = (k % 2 === 0) ? 'even' : 'odd';
      this.createPoint(this.fields[0], k, typeClass);
    }

    // Bottom left field (11-6)
    for (var k = 11; k >= 6; k--) {
      var typeClass = (k % 2 === 0) ? 'even' : 'odd';
      this.createPoint(this.fields[1], k, typeClass);
    }

    // Top right field (18-23)
    for (var k = 18; k <= 23; k++) {
      var typeClass = (k % 2 === 0) ? 'even' : 'odd';
      this.createPoint(this.fields[2], k, typeClass);
    }

    // Bottom right field (5-0)
    for (var k = 5; k >= 0; k--) {
      var typeClass = (k % 2 === 0) ? 'even' : 'odd';
      this.createPoint(this.fields[3], k, typeClass);
    }
  },

  /**
   * Create a piece element
   *
   * @param {jQuery} parentElem - Parent element to add piece to
   * @param {Piece} piece - Piece object
   * @param {number} height - Optional height offset
   */
  createPiece: function(parentElem, piece, height) {
    var pieceElem = $(
      '<div id="piece' + piece.id + '" class="piece ' + piece.type + '">' +
      '<div class="image"></div>' +
      '</div>'
    );

    pieceElem.data('piece', piece);
    if (height) {
      pieceElem.data('boostedHeight', height);
    }
    parentElem.append(pieceElem);
  },

  /**
   * Compact pieces in all positions
   */
  compactAllPositions: function() {
    // Compact all points
    for (var i = 0; i < 24; i++) {
      this.compactPosition(i);
    }

    // Compact bar
    this.compactElement(
      this.getBarElem(model.PieceType.WHITE),
      this.client.player.currentPieceType === model.PieceType.WHITE ? 'top' : 'bottom'
    );
    this.compactElement(
      this.getBarElem(model.PieceType.BLACK),
      this.client.player.currentPieceType === model.PieceType.BLACK ? 'top' : 'bottom'
    );
  },

  /**
   * Compact pieces in specific DOM element to make them fit vertically.
   *
   * @param {jQuery} element - DOM element containing pieces
   * @param {string} alignment - Alignment of pieces - 'top' or 'bottom'
   */
  compactElement: function(element, alignment) {
    var negAlignment = alignment === 'top' ? 'bottom' : 'top';
    var elementHeight = element.height();
    var itemCount = element.children().length;

    if (itemCount === 0) {
      return;
    }

    var firstItem = element.children().first();
    var itemHeight = firstItem.width();

    // Calculate margin percentage based on item height
    var maxItemsVisible = Math.floor(elementHeight / itemHeight);
    var marginPercent = 100;

    if (itemCount > maxItemsVisible) {
      marginPercent = 100 - ((itemHeight * maxItemsVisible / itemCount) / itemHeight) * 100;
    }

    // Apply margins to each item
    element.children().each(function(i) {
      var boostedChipRank = $(this).data('boostedHeight') || 0;

      // For pieces with boosted height, position them differently
      if (boostedChipRank > 0) {
        boostedChipRank = Math.min(boostedChipRank, 6);

        // Place on top of other pieces with adjusted z-index
        $(this).css("z-index", 1001);
        $(this).children().first().text("(" + boostedChipRank + ")");
      } else {
        // Normal positioning for regular pieces
        $(this).css("z-index", "auto");
        $(this).children().first().text("");
      }

      // Set positioning based on alignment
      $(this).css(alignment, "0");
      $(this).css("margin-" + alignment, self.toFixedDown(marginPercent, 2) + "%");
      $(this).css(negAlignment, "inherit");
      $(this).css("margin-" + negAlignment, "inherit");
    });
  },

  /**
   * Compact pieces in specific position to make them fit on screen vertically.
   *
   * @param {Number} pos - Position of point
   */
  compactPosition: function(pos) {
    var pointElement = this.getPointElem(pos);
    var alignment = (pos >= 12) ? 'top' : 'bottom';
    this.compactElement(pointElement, alignment);
  },

  /**
   * Create pieces on the board based on game state
   */
  createPieces: function() {
    var game = this.match.currentGame;

    // Check if we have a board array from the API or state points
    if (game.board) {
      // Create pieces from board array (API format)
      this.createPiecesFromBoardArray(game.board);
    } else if (game.state && game.state.points) {
      // Create pieces from state points (original backgammon.js format)
      var state = game.state;

      // Create pieces on points
      for (var pos = 0; pos < 24; pos++) {
        var point = state.points[pos];
        for (var i = 0; i < point.length; i++) {
          var pointElem = this.getPointElem(pos);
          this.createPiece(pointElem, point[i], 0);
        }
        this.compactPosition(pos);
      }

      // Create pieces on bar
      for (var i = 0; i < state.bar.white.length; i++) {
        var piece = state.bar.white[i];
        var barElem = this.getBarElem(piece.type);
        this.createPiece(barElem, piece, 0);
      }

      for (var i = 0; i < state.bar.black.length; i++) {
        var piece = state.bar.black[i];
        var barElem = this.getBarElem(piece.type);
        this.createPiece(barElem, piece, 0);
      }
    } else {
      console.error('No valid board data found in game state');
    }

    // Compact all positions
    this.compactAllPositions();
  },

  /**
   * Create pieces from a board array (API format)
   * The board array contains values representing checkers:
   * - Positive values: White checkers (count)
   * - Negative values: Black checkers (count)
   * - 0: Empty position
   *
   * @param {Array} boardArray - Board array from API
   */
  createPiecesFromBoardArray: function(boardArray) {
    console.log('Creating pieces from board array:', boardArray);

    // Create a piece ID counter for each type
    var pieceIdCounter = {
      white: 1,
      black: 1
    };

    // Loop through all positions
    for (var pos = 0; pos < 24; pos++) {
      var value = boardArray[pos];

      if (value !== 0) {
        // Determine piece type and count
        var type = value > 0 ? model.PieceType.WHITE : model.PieceType.BLACK;
        var count = Math.abs(value);

        // Get the point element
        var pointElem = this.getPointElem(pos);

        // Create pieces at this position
        for (var i = 0; i < count; i++) {
          // Create a piece object
          var piece = new model.Piece(pieceIdCounter[type]++, type);

          // Create the piece element
          this.createPiece(pointElem, piece, 0);
        }

        // Compact this position
        this.compactPosition(pos);
      }
    }

    console.log('Pieces created from board array');
  },

  /**
   * Remove all pieces from the board
   */
  removePieces: function() {
    // Remove pieces from points
    for (var pos = 0; pos < 24; pos++) {
      var pointElem = this.getPointElem(pos);
      pointElem.empty();
    }

    // Remove pieces from bar
    this.getBarElem(model.PieceType.BLACK).empty();
    this.getBarElem(model.PieceType.WHITE).empty();
  },

  /**
   * Remove all points from the board
   */
  removePoints: function() {
    for (var i = 0; i < 4; i++) {
      this.fields[i].empty();
    }
  },

  /**
   * Reset board UI
   *
   * @param {Match} match - Match object
   * @param {Rule} rule - Rule object
   */
  resetBoard: function(match, rule) {
    this.match = match;
    this.rule = rule;

    console.log('Resetting board with match:', match);

    // If points don't exist yet, create them
    if (this.getPointElem(0).length === 0) {
      this.removePoints();
      this.createPoints();
    }

    // Remove existing pieces and create new ones
    this.removePieces();
    this.createPieces();

    this.randomizeDiceRotation();

    this.updateControls();
    this.updateScoreboard();
    this.compactAllPositions();

    console.log('Board reset complete');
  },

  /**
   * Notify that player undid move
   */
  notifyUndo: function() {
    this.notifyInfo('Player undid last move.');
  },

  /**
   * Generate random rotation angles for dice
   */
  randomizeDiceRotation: function() {
    for (var i = 0; i < 2; i++) {
      this.rotationAngle[i] = Math.random() * 30 - 15;
    }
  },

  /**
   * Update controls based on game state
   */
  updateControls: function() {
    var game = this.match.currentGame;

    console.log('Updating controls based on game state:', game);

    // Hide all action buttons first
    $('#btn-roll').hide();
    $('#btn-confirm').hide();
    $('#btn-undo').hide();
    $('#menu-resign').hide();
    $('#menu-undo').hide();

    // Default show state for buttons
    var canRoll = false;
    var canConfirmMove = false;
    var canUndoMove = false;

    if (game) {
      // Determine if this is player's turn
      var isPlayerTurn = game.turnPlayer &&
                         this.client.player &&
                         game.turnPlayer.currentPieceType === this.client.player.currentPieceType;

      // Check if dice was rolled
      var diceWasRolled = game.turnDice !== null;

      // Check if there are more moves
      var hasMoreMoves = game.turnDice && game.turnDice.movesLeft && game.turnDice.movesLeft.length > 0;

      // Check if there are pending actions
      var hasPendingActions = game.pendingActions && game.pendingActions.length > 0;

      // Adjusted conditions for buttons
      canRoll = isPlayerTurn &&
                !diceWasRolled &&
                game.hasStarted &&
                !game.isOver;

      canConfirmMove = isPlayerTurn &&
                      diceWasRolled &&
                      !hasMoreMoves &&
                      !game.isOver;

      // For simplicity in the initial implementation, allow undo even without pending actions
      canUndoMove = isPlayerTurn && diceWasRolled && !game.isOver;

      console.log('Button states - canRoll:', canRoll, 'canConfirmMove:', canConfirmMove, 'canUndoMove:', canUndoMove);
    }

    // Show appropriate buttons
    $('#btn-roll').toggle(canRoll);
    $('#btn-confirm').toggle(canConfirmMove);
    $('#btn-undo').toggle(canUndoMove);
    $('#menu-resign').toggle(game && game.hasStarted && !game.isOver);
    $('#menu-undo').toggle(canUndoMove);

    // Update dice display
    if (game && game.turnDice) {
      this.updateDice(game.turnDice, game.turnPlayer.currentPieceType);
    } else {
      $('.dice-panel').hide();
    }

    console.log('Controls updated');
  },

  /**
   * Update the board UI to match current game state
   */
  updateBoard: function() {
    console.log('Board UI updated');

    var game = this.match.currentGame;

    console.log('Match:', this.match);
    console.log('Game:', game);
    console.log('Player:', this.client.player);

    this.removePieces();
    this.createPieces();
    this.updateControls();
    this.updateScoreboard();
    this.compactAllPositions();
  },

  /**
   * Update the scoreboard
   */
  updateScoreboard: function() {
    var match = this.match;
    var game = match.currentGame;

    // Update match state
    var matchText = "Not in a match";
    var matchTextTitle = "";

    if (match) {
      if (match.currentGameNumber > 0) {
        matchText = "Game " + match.currentGameNumber + " of " + match.gameCount;
        matchTextTitle = "First to " + match.pointsToWin + " points wins the match";
      }
    }

    $('#match-state').text(matchText);
    $('#match-state').attr('title', matchTextTitle);

    // Update score
    var yourscore = 0;
    var oppscore = 0;

    if (match && match.score) {
      if (this.client.player.currentPieceType === model.PieceType.WHITE) {
        yourscore = match.score.white || 0;
        oppscore = match.score.black || 0;
      } else {
        yourscore = match.score.black || 0;
        oppscore = match.score.white || 0;
      }
    }

    $('#yourscore').text(yourscore);

    if (match && match.isStarted) {
      $('#oppscore').text(oppscore);
    } else {
      $('#oppscore').text('');
    }
  },

  /**
   * Show game result overlay
   *
   * @param {string} message - Result message
   * @param {string} matchState - Match state message
   * @param {string} color - Color of the message
   */
  showGameResult: function(message, matchState, color) {
    var match = this.match;
    var yourscore = 0;
    var oppscore = 0;

    if (match && match.score) {
      if (this.client.player.currentPieceType === model.PieceType.WHITE) {
        yourscore = match.score.white || 0;
        oppscore = match.score.black || 0;
      } else {
        yourscore = match.score.black || 0;
        oppscore = match.score.white || 0;
      }
    }

    $('#game-result-overlay').show();

    $('.game-result').css('color', color);
    $('.game-result .message').html(message);
    $('.game-result .state').html(matchState);

    $('.game-result .yourscore').text(yourscore);
    $('.game-result .oppscore').text(oppscore);
    $('.game-result .text').each(function() {
      if (window.fitText) {
        window.fitText($(this));
      }
    });
  },

  /**
   * Show player resigned message
   */
  showResigned: function() {
    this.notifyInfo('Other player resigned from game');
  },

  /**
   * Updates the DOM element representing the specified die (specified by index).
   * Changes CSS styles of the element.
   *
   * @param {Dice} dice - Dice to render
   * @param {number} index - Index of dice value in array
   * @param {PieceType} type - Player's type
   */
  updateDie: function(dice, index, type) {
    var color = (type === model.PieceType.WHITE) ? 'white' : 'black';
    var id = '#die' + index;

    // Ensure the die element exists
    if ($(id).length === 0) {
      console.log('Creating die element:', id);
      var diceContainer = (type === this.client.player.currentPieceType) ? $('#dice-right') : $('#dice-left');
      diceContainer.append('<span id="die' + index + '" class="die"></span>');
    }

    // Set text (as fallback)
    $(id).text(dice.values[index]);

    // Change image based on value
    $(id).removeClass('digit-1-white digit-2-white digit-3-white digit-4-white digit-5-white digit-6-white digit-1-black digit-2-black digit-
-black digit-4-black digit-5-black digit-6-black played');
    $(id).addClass('digit-' + dice.values[index] + '-' + color);

    // Mark as played if already used
    if (dice.movesLeft && dice.movesLeft.indexOf(dice.values[index]) === -1) {
      $(id).addClass('played');
    }

    // Set rotation for realistic look
    var angle = this.rotationAngle[index];
    $(id).css('transform', 'rotate(' + angle + 'deg)');

    console.log('Die updated:', id, 'value:', dice.values[index], 'color:', color);
  },

  /**
   * Update dice display
   *
   * @param {Dice} dice - Dice object
   * @param {PieceType} type - Player's type
   */
  updateDice: function(dice, type) {
    console.log('Updating dice display:', dice, 'type:', type);

    // Make dice panels visible
    $('.dice-panel').show();

    // Clear existing dice
    $('.dice').each(function() {
      $(this).empty();
    });

    // We always show dice in right pane for current player
    // and in left pane for opponent
    var diceElem;
    if (type === this.client.player.currentPieceType) {
      diceElem = $('#dice-right');
    } else {
      diceElem = $('#dice-left');
    }

    // Create and update dice
    for (var i = 0; i < dice.values.length; i++) {
      diceElem.append('<span id="die' + i + '" class="die"></span>');
      this.updateDie(dice, i, type);
    }

    // Enable clicking on dice to rotate values (for the player's dice only)
    if (type === this.client.player.currentPieceType) {
      var self = this;
      $('.dice .die').unbind('click');
      $('.dice .die').click(function() {
        console.log('Dice clicked - rotating values:', dice.values);
        console.log('Moves left before rotation:', dice.movesLeft);

        // Only allow rotating if we have moves left
        if (dice.movesLeft && dice.movesLeft.length > 0) {
          model.Utils.rotateLeft(dice.values);
          model.Utils.rotateLeft(dice.movesLeft);

          console.log('Moves left after rotation:', dice.movesLeft);

          for (var i = 0; i < dice.values.length; i++) {
            self.updateDie(dice, i, type);
          }

          self.updateControls();
        } else {
          console.log('No moves left to rotate');
        }
      });
    }

    console.log('Dice display updated');
  },

  /**
   * Play a move action
   *
   * @param {Action} action - Move action to play
   */
  playMoveAction: function(action) {
    if (!action.piece) {
      throw new Error('No piece!');
    }

    var pieceElem = this.getPieceElem(action.piece);
    var srcPointElem = pieceElem.parent();
    var dstPointElem = this.getPointElem(action.to);

    pieceElem.detach();
    dstPointElem.append(pieceElem);

    this.compactPosition(srcPointElem.data('position'));
    this.compactPosition(dstPointElem.data('position'));
  },

  /**
   * Play a recover action (piece from bar)
   *
   * @param {Action} action - Recover action to play
   */
  playRecoverAction: function(action) {
    if (!action.piece) {
      throw new Error('No piece!');
    }

    var pieceElem = this.getPieceElem(action.piece);
    var srcPointElem = pieceElem.parent();
    var dstPointElem = this.getPointElem(action.position);

    pieceElem.detach();
    dstPointElem.append(pieceElem);

    this.compactElement(srcPointElem, action.piece.type === this.client.player.currentPieceType ? 'top' : 'bottom');
    this.compactPosition(dstPointElem.data('position'));
  },

  /**
   * Play a hit action (piece to bar)
   *
   * @param {Action} action - Hit action to play
   */
  playHitAction: function(action) {
    if (!action.piece) {
      throw new Error('No piece!');
    }

    var pieceElem = this.getPieceElem(action.piece);
    var srcPointElem = pieceElem.parent();
    var dstPointElem = this.getBarElem(action.piece.type);

    pieceElem.detach();
    dstPointElem.append(pieceElem);

    this.compactPosition(srcPointElem.data('position'));
    this.compactElement(dstPointElem, action.piece.type === this.client.player.currentPieceType ? 'top' : 'bottom');
  },

  /**
   * Play a bear action (piece off the board)
   *
   * @param {Action} action - Bear action to play
   */
  playBearAction: function(action) {
    if (!action.piece) {
      throw new Error('No piece!');
    }

    var pieceElem = this.getPieceElem(action.piece);
    var srcPointElem = pieceElem.parent();

    pieceElem.detach();

    this.compactPosition(srcPointElem.data('position'));
  },

  /**
   * Play an up action (piece height adjustment)
   *
   * @param {Action} action - Up action to play
   */
  playUpAction: function(action) {
    if (!action.piece) {
      throw new Error('No piece!');
    }

    var pieceElem = this.getPieceElem(action.piece);
    pieceElem.data('boostedHeight', action.to);

    this.compactPosition(action.from);
  },

  /**
   * Compact pieces after UI was resized
   */
  resize: function() {
    this.compactAllPositions();
  }
};

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SimpleBoardUI;
}
