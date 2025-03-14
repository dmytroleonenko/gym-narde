document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM content loaded, initializing game...');
    
    // Create move validator
    const moveValidator = new MoveValidator();
    
    // Game state
    let gameState = {
        board: Array(24).fill(0),
        currentPlayer: null,
        dice: [1, 1],
        validMovesByPiece: {},
        firstMoveMade: false,
        selectedPiece: null,
        validDestinations: [],
        gameId: null,
        borneOff: {
            white: 0,
            black: 0
        },
        isFirstTurn: true,
        headMoveMade: false,
        headMovesMade: 0,
        HEAD_POSITIONS: {
            'white': 23,
            'black': 11
        },
        usedDice: [] // Track which dice have been used
    };

    // DOM elements
    const newGameBtn = document.getElementById('new-game-btn');
    const feedbackBtn = document.getElementById('feedback-btn');
    const shutdownBtn = document.getElementById('shutdown-btn');
    const gameStatus = document.getElementById('game-status');
    const board = document.getElementById('board');
    const piecesContainer = document.getElementById('pieces-container');
    const dice1 = document.getElementById('dice1');
    const dice2 = document.getElementById('dice2');
    const whiteOffCount = document.getElementById('white-borne-off');
    const blackOffCount = document.getElementById('black-borne-off');
    const whiteOffContainer = document.getElementById('white-borne-off-container');
    const blackOffContainer = document.getElementById('black-borne-off-container');
    const modal = document.getElementById('message-modal');
    const modalMessage = document.getElementById('modal-message');
    const closeModal = document.getElementById('close-modal');
    
    // Feedback modal elements
    const feedbackModal = document.getElementById('feedback-modal');
    const feedbackText = document.getElementById('feedback-text');
    const cancelFeedback = document.getElementById('cancel-feedback');
    const saveFeedback = document.getElementById('save-feedback');
    const submitFeedback = document.getElementById('submit-feedback');
    
    // Debug: Check if the elements were found 
    console.log('Feedback modal elements:');
    console.log(' - feedbackModal:', feedbackModal);
    console.log(' - feedbackText:', feedbackText);
    console.log(' - cancelFeedback:', cancelFeedback);
    console.log(' - saveFeedback:', saveFeedback);
    console.log(' - submitFeedback:', submitFeedback);

    // Start a new game
    newGameBtn.addEventListener('click', startNewGame);
    
    // Show feedback modal (without shutting down)
    if (feedbackBtn) {
        feedbackBtn.addEventListener('click', function() {
            console.log('Feedback button clicked');
            if (feedbackModal) {
                feedbackModal.style.display = 'block';
            } else {
                console.error('Feedback modal element not found!');
                alert('Feedback form not available.');
            }
        });
    } else {
        console.error('Feedback button not found');
    }
    
    // Shutdown server with feedback
    shutdownBtn.addEventListener('click', function() {
        console.log('Shutdown button clicked');
        console.log('Feedback modal element:', feedbackModal);
        
        // Show feedback modal instead of immediate shutdown
        if (feedbackModal) {
            console.log('Setting feedbackModal display to block');
            feedbackModal.style.display = 'block';
        } else {
            console.error('Feedback modal element not found!');
            alert('Feedback form not available. Server will shut down without feedback.');
            
            // Fallback to direct shutdown
            fetch('/shutdown', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ feedback: 'Feedback modal not available' })
            });
        }
    });
    
    // Cancel feedback and close modal
    if (cancelFeedback) {
        cancelFeedback.addEventListener('click', function() {
            console.log('Cancel feedback clicked');
            feedbackModal.style.display = 'none';
        });
    } else {
        console.error('Cancel feedback button not found');
    }
    
    // Submit feedback and shutdown
    // Save feedback without shutting down
    if (saveFeedback) {
        saveFeedback.addEventListener('click', function() {
            console.log('Save feedback clicked');
            const feedback = feedbackText ? feedbackText.value.trim() : '';
            console.log('Feedback text:', feedback);
            
            fetch('/api/save_feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    feedback: feedback
                })
            }).then(response => response.json())
            .then(data => {
                console.log('Feedback saved successfully:', data);
                alert('Thank you for your feedback!');
                feedbackModal.style.display = 'none';
            }).catch(error => {
                console.error('Error saving feedback:', error);
                alert('There was an error saving your feedback. Please try again.');
            });
        });
    } else {
        console.error('Save feedback button not found');
    }
    
    // Submit feedback and shutdown
    if (submitFeedback) {
        submitFeedback.addEventListener('click', function() {
            console.log('Submit feedback clicked');
            const feedback = feedbackText ? feedbackText.value.trim() : '';
            console.log('Feedback text:', feedback);
            
            fetch('/shutdown', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    feedback: feedback
                })
            }).then(response => {
                console.log('Feedback submitted successfully');
                alert('Thank you for your feedback! Server is shutting down...');
            }).catch(error => {
                console.error('Error sending feedback:', error);
                alert('There was an error sending your feedback, but the server will shut down.');
            });
        });
    } else {
        console.error('Submit feedback button not found');
    }

    // Close modal
    closeModal.addEventListener('click', function() {
        modal.style.display = 'none';
    });

    // Start a new game
    function startNewGame() {
        fetch('/api/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            gameState = {
                board: data.board,
                currentPlayer: data.current_player,
                dice: data.dice,
                validMovesByPiece: data.valid_moves_by_piece || {},
                firstMoveMade: false,
                selectedPiece: null,
                validDestinations: [],
                gameId: data.game_id,
                borneOff: data.borne_off,
                isFirstTurn: true,
                headMoveMade: false,
                headMovesMade: 0,
                HEAD_POSITIONS: {
                    'white': 23,
                    'black': 11
                },
                usedDice: [] // Reset used dice for new game
            };
            
            // Initialize move validator with new game state
            moveValidator.initialize(
                gameState.board,
                gameState.dice,
                gameState.currentPlayer,
                true, // isFirstTurn
                false // headMoveMade
            );
            
            renderDice();
            renderBoard();
            renderBorneOff();
            updateGameStatus('Your turn (White) - Roll: ' + gameState.dice.join(', '));
        })
        .catch(error => {
            console.error('Error starting game:', error);
            showModal('Error starting game. Please try again.');
        });
    }

    // Render the dice
    function renderDice() {
        dice1.textContent = gameState.dice[0];
        dice2.textContent = gameState.dice[1];
    }

    // Render borne off pieces
    function renderBorneOff() {
        whiteOffCount.textContent = gameState.borneOff.white;
        blackOffCount.textContent = gameState.borneOff.black;
        
        // Clear containers
        whiteOffContainer.innerHTML = '';
        blackOffContainer.innerHTML = '';
        
        // Add white borne off pieces
        for (let i = 0; i < gameState.borneOff.white; i++) {
            const piece = document.createElement('div');
            piece.className = 'borne-off-piece white';
            whiteOffContainer.appendChild(piece);
        }
        
        // Add black borne off pieces
        for (let i = 0; i < gameState.borneOff.black; i++) {
            const piece = document.createElement('div');
            piece.className = 'borne-off-piece black';
            blackOffContainer.appendChild(piece);
        }
    }

    // Apply head rule to valid moves
    function applyHeadRuleToValidMoves() {
        console.log("Applying head rule to valid moves...");
        
        // NUCLEAR OPTION: If we have made any head move, completely block position 23
        if (gameState.headMoveMade || gameState.headMovesMade > 0 || window.headMoveBlockActive) {
            console.log("🚫 NUCLEAR BLOCKING: Force removing head position 23 from valid moves");
            
            // Remove head position from valid moves
            if (gameState.validMovesByPiece && gameState.validMovesByPiece[23]) {
                console.log(`🚫 Removed head position 23 from valid moves`);
                delete gameState.validMovesByPiece[23];
            }
            
            // Set the global block flag
            window.headMoveBlockActive = true;
        }
    }

    // Render the board
    function renderBoard() {
        // Apply head rule filtering before rendering
        applyHeadRuleToValidMoves();
        
        // Clear pieces container
        piecesContainer.innerHTML = '';
        
        // Reset all point classes
        document.querySelectorAll('.point').forEach(point => {
            point.classList.remove('valid-destination');
        });
        
        // Loop through all positions and create pieces
        for (let position = 0; position < 24; position++) {
            const count = gameState.board[position];
            if (count !== 0) {
                // Create a piece at this position
                const piece = document.createElement('div');
                const isWhite = count > 0;
                const absCount = Math.abs(count);
                
                piece.className = `piece ${isWhite ? 'white' : 'black'}`;
                
                // Store position using 0-based index for API calls
                piece.setAttribute('data-position', position);
                piece.setAttribute('data-count', absCount);
                
                // Position the piece on the board
                const pointElement = document.getElementById(`point-${position}`);
                const rect = pointElement.getBoundingClientRect();
                const boardRect = board.getBoundingClientRect();
                
                // Calculate position (centered in point)
                let left = pointElement.offsetLeft + (pointElement.offsetWidth - 40) / 2;
                let top;
                
                // Adjust for stacking
                if (position >= 12) {
                    // Top row - pieces stack down
                    top = 10;
                } else {
                    // Bottom row - pieces stack up
                    top = board.offsetHeight - 50;
                }
                
                piece.style.left = `${left}px`;
                piece.style.top = `${top}px`;
                
                // Add stack count if more than 1
                if (absCount > 1) {
                    const stackCount = document.createElement('div');
                    stackCount.className = 'stack-count';
                    stackCount.textContent = absCount;
                    piece.appendChild(stackCount);
                }
                
                // Add event listeners for drag and drop only for player's pieces
                if (isWhite && gameState.currentPlayer === 'white') {
                    // DIRECT HEAD RULE CHECK: Prevent head moves after first one
                    let canMove = position in gameState.validMovesByPiece;
                    
                    // Explicitly check head rule
                    if (position === gameState.HEAD_POSITIONS?.white && gameState.headMoveMade) {
                        console.log(`BLOCKED HEAD MOVE in renderBoard: Cannot make second move from head position ${position}`);
                        canMove = false;
                    }
                    
                    // Additional debug to help identify why pieces aren't showing as movable
                    console.log(`validMovesByPiece keys: ${Object.keys(gameState.validMovesByPiece).join(', ')}`);
                    
                    // Log all pieces and their move status for debugging
                    console.log(`Piece at position ${position}, canMove: ${canMove}, count: ${absCount}`);
                    
                    // Log to server the current game state (debug)
                    fetch('/api/log', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: `Client: All validMovesByPiece: ${JSON.stringify(gameState.validMovesByPiece)}`
                        })
                    });
                    
                    if (canMove) {
                        piece.classList.add('valid-move');
                        console.log(`Piece at position ${position} can move to:`, gameState.validMovesByPiece[position]);
                        
                        // Log to server
                        fetch('/api/log', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                message: `Client: Piece at position ${position} can move to: ${JSON.stringify(gameState.validMovesByPiece[position])}`
                            })
                        });
                        
                        // Mouse down - start drag
                        piece.addEventListener('mousedown', startDrag);
                        
                        // Touch start for mobile
                        piece.addEventListener('touchstart', startDrag, { passive: false });
                    }
                }
                
                piecesContainer.appendChild(piece);
            }
        }
    }

    // Start dragging a piece
    function startDrag(e) {
        e.preventDefault();
        
        // Get position data
        const position = parseInt(this.getAttribute('data-position'));
        
        // DIRECT CHECK: Hard-block the head position if already made a head move
        if (position === gameState.HEAD_POSITIONS?.white && (gameState.headMoveMade || gameState.headMovesMade > 0)) {
            console.log("BLOCKED HEAD MOVE in startDrag: Cannot make second move from head position");
            showModal("Only one checker may leave the head position per turn");
            return;
        }
        
        // Select this piece
        gameState.selectedPiece = position;
        this.classList.add('dragging');
        
        // Get valid destinations for this piece
        getValidMoves(position);
        
        // Set up move and end events
        document.addEventListener('mousemove', moveDrag);
        document.addEventListener('mouseup', endDrag);
        document.addEventListener('touchmove', moveDrag, { passive: false });
        document.addEventListener('touchend', endDrag);
        
        // Record initial mouse/touch position
        const clientX = e.clientX || e.touches[0].clientX;
        const clientY = e.clientY || e.touches[0].clientY;
        
        // Store offset within the piece where the drag started
        this.setAttribute('data-offset-x', clientX - this.getBoundingClientRect().left);
        this.setAttribute('data-offset-y', clientY - this.getBoundingClientRect().top);
        
        // Move piece to initial position
        moveDrag(e);
    }

    // Move piece during drag
    function moveDrag(e) {
        e.preventDefault();
        
        const draggedPiece = document.querySelector('.piece.dragging');
        if (!draggedPiece) return;
        
        const clientX = e.clientX || (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
        const clientY = e.clientY || (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
        
        const offsetX = parseInt(draggedPiece.getAttribute('data-offset-x')) || 20;
        const offsetY = parseInt(draggedPiece.getAttribute('data-offset-y')) || 20;
        
        const boardRect = board.getBoundingClientRect();
        const left = clientX - boardRect.left - offsetX;
        const top = clientY - boardRect.top - offsetY;
        
        // Update piece position
        draggedPiece.style.left = `${left}px`;
        draggedPiece.style.top = `${top}px`;
    }

    // End dragging a piece
    function endDrag(e) {
        e.preventDefault();
        
        // Remove event listeners
        document.removeEventListener('mousemove', moveDrag);
        document.removeEventListener('mouseup', endDrag);
        document.removeEventListener('touchmove', moveDrag);
        document.removeEventListener('touchend', endDrag);
        
        const draggedPiece = document.querySelector('.piece.dragging');
        if (!draggedPiece) return;
        
        // Get position of dropped piece
        const fromPosition = parseInt(draggedPiece.getAttribute('data-position'));
        
        // Get drop target
        const clientX = e.clientX || (e.changedTouches && e.changedTouches[0] ? e.changedTouches[0].clientX : 0);
        const clientY = e.clientY || (e.changedTouches && e.changedTouches[0] ? e.changedTouches[0].clientY : 0);
        
        // Check if dropping on bearing off area
        const bearingOffRect = document.getElementById('bearing-off').getBoundingClientRect();
        if (clientX >= bearingOffRect.left && clientX <= bearingOffRect.right && 
            clientY >= bearingOffRect.top && clientY <= bearingOffRect.bottom) {
            
            // Check if bearing off is a valid move
            if (gameState.validDestinations.includes(-1)) {
                makeMove(fromPosition, -1);
                return;
            }
        }
        
        // Check if dropping on valid point
        let validDrop = false;
        for (const toPosition of gameState.validDestinations) {
            if (toPosition === -1) continue;  // Skip 'off' for point check
            
            const pointElement = document.getElementById(`point-${toPosition}`);
            const rect = pointElement.getBoundingClientRect();
            
            if (clientX >= rect.left && clientX <= rect.right && 
                clientY >= rect.top && clientY <= rect.bottom) {
                
                // Valid drop - make the move
                makeMove(fromPosition, toPosition);
                validDrop = true;
                break;
            }
        }
        
        // Invalid drop - return piece to original position
        if (!validDrop) {
            draggedPiece.classList.remove('dragging');
            renderBoard();
        }
    }

    // Get valid moves for a piece
    function getValidMoves(position) {
        // This is a nuclear option to enforce the head rule
        // If we EVER try to get moves for position 23 after a head move, we block it
        if (position === 23) {
            // Use a local variable to track head moves during this function call
            const headMoveCount = gameState.headMovesMade || 0;
            const headMoveMadeFlag = gameState.headMoveMade || false;
            console.log(`NUCLEAR HEAD RULE CHECK: position=${position}, headMoveMade=${headMoveMadeFlag}, headMovesMade=${headMoveCount}`);
            
            if (headMoveMadeFlag) {
                console.log("🚫 NUCLEAR OPTION: Blocking all moves from head position 23 after first head move");
                showModal("Only one checker may leave the head position per turn");
                gameState.validDestinations = [];
                highlightValidDestinations([]);
                return;
            }
        }
        
        // Always check if we have used dice state that we should track
        const usedDice = gameState.usedDice || [];
        const remainingDice = [...gameState.dice]; // Start with original dice
        
        // Remove used dice from remaining
        usedDice.forEach(dieValue => {
            const index = remainingDice.indexOf(dieValue);
            if (index !== -1) {
                remainingDice.splice(index, 1);
            }
        });
        
        console.log('Getting valid moves with dice state:', {
            original: gameState.dice,
            remaining: remainingDice,
            used: usedDice
        });
        
        // Initialize client-side validation with current dice state
        console.log(`Initializing MoveValidator with headMoveMade=${gameState.headMoveMade}, headMovesMade=${gameState.headMovesMade}`);
        
        moveValidator.initialize(
            gameState.board, 
            gameState.dice, // Always initialize with the original dice
            gameState.currentPlayer,
            gameState.isFirstTurn,
            gameState.headMovesMade > 0 // Use headMovesMade count to determine if a head move was made
        );
        
        // Update dice state to match our tracking
        if (usedDice.length > 0) {
            // Manually update the dice state to remove used dice
            usedDice.forEach(dieValue => {
                moveValidator.updateDiceAfterMove(dieValue);
            });
        }
        
        // Get valid destinations using client-side validator
        gameState.validDestinations = moveValidator.getValidMovesForPosition(position);
        console.log(`Client-side valid destinations for ${position}: ${gameState.validDestinations.join(', ')}`);
        
        // Display valid destinations immediately
        highlightValidDestinations(gameState.validDestinations);
        
        // Still get server validation as backup and to keep in sync
        // NUCLEAR OPTION: For position 23, don't even ask the server if we've already made a head move
        if (position === 23 && (gameState.headMoveMade || gameState.headMovesMade > 0 || window.headMoveBlockActive)) {
            console.log("🚫 NUCLEAR BLOCKING: Skipping server query for position 23 after head move");
            return;
        }
        
        fetch('/api/get_valid_moves', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ position: position })
        })
        .then(response => response.json())
        .then(data => {
            // NUCLEAR OPTION: For position 23, ignore server response if we've already made a head move
            if (position === 23 && (gameState.headMoveMade || gameState.headMovesMade > 0 || window.headMoveBlockActive)) {
                console.log("🚫 NUCLEAR BLOCKING: Ignoring server response for position 23 after head move");
                gameState.validDestinations = [];
                highlightValidDestinations([]);
                return;
            }
            
            // Update with server response
            const serverDestinations = data.valid_to_positions || [];
            console.log(`Server-side valid destinations for ${position}: ${serverDestinations.join(', ')}`);
            
            // Use server's response for final validation
            gameState.validDestinations = serverDestinations;
            
            // Update highlights with server response
            highlightValidDestinations(serverDestinations);
        })
        .catch(error => {
            console.error('Error getting valid moves from server:', error);
            // We still have client-side validation so can continue
        });
    }
    
    // Helper function to highlight valid destinations
    function highlightValidDestinations(destinations) {
        // Highlight valid destinations
        document.querySelectorAll('.point').forEach(point => {
            point.classList.remove('valid-destination');
        });
        
        destinations.forEach(pos => {
            if (pos !== -1) {  // Skip 'off' for highlighting
                const pointElement = document.getElementById(`point-${pos}`);
                if (pointElement) {
                    pointElement.classList.add('valid-destination');
                }
            }
        });
        
        // Highlight bearing off area if it's a valid destination
        const bearingOff = document.getElementById('bearing-off');
        if (destinations.includes(-1)) {
            bearingOff.classList.add('valid-destination');
        } else {
            bearingOff.classList.remove('valid-destination');
        }
    }

    // Make a move
    function makeMove(fromPosition, toPosition) {
        // Calculate which die is being used for this move
        const dieValue = moveValidator.getMoveDistance(fromPosition, toPosition);
        console.log(`Move from ${fromPosition} to ${toPosition} uses die value: ${dieValue}`);
        
        // Initialize the validator with existing dice state
        const usedDice = gameState.usedDice || [];
        const remainingDice = [...gameState.dice]; // Start with original dice
        
        // Remove used dice from remaining
        usedDice.forEach(dieValue => {
            const index = remainingDice.indexOf(dieValue);
            if (index !== -1) {
                remainingDice.splice(index, 1);
            }
        });
        
        // Initialize with current state
        console.log(`Initializing MoveValidator in makeMove with headMoveMade=${gameState.headMoveMade}, headMovesMade=${gameState.headMovesMade}`);
        
        moveValidator.initialize(
            gameState.board, 
            gameState.dice, 
            gameState.currentPlayer,
            gameState.isFirstTurn,
            gameState.headMovesMade > 0 // Use headMovesMade count to determine if a head move was made
        );
        
        // Apply any previously used dice
        usedDice.forEach(usedDieValue => {
            moveValidator.updateDiceAfterMove(usedDieValue);
        });
        
        // Validate the current move
        // Check head rule directly before validation
        if (fromPosition === gameState.HEAD_POSITIONS?.white) {
            // Only allow 2 head moves if it's first turn special doubles
            const isFirstTurnSpecialDoubles = 
                gameState.isFirstTurn && 
                gameState.dice && 
                gameState.dice.length >= 2 && 
                gameState.dice[0] === gameState.dice[1] && 
                [3, 4, 6].includes(gameState.dice[0]);
                
            // Log the head move tracking for debugging
            console.log(`Head move check: headMoveMade=${gameState.headMoveMade}, headMovesMade=${gameState.headMovesMade}, isFirstTurnSpecialDoubles=${isFirstTurnSpecialDoubles}`);
                
            if (gameState.headMovesMade >= (isFirstTurnSpecialDoubles ? 2 : 1)) {
                showModal("Only one checker may leave the head position per turn");
                renderBoard();
                return;
            }
        }
            
        const validation = moveValidator.validateMove(fromPosition, toPosition);
        
        // If invalid according to client, show error and return early
        if (!validation.valid) {
            showModal(validation.error);
            renderBoard();
            return;
        }
        
        // If valid according to client, send to server for final validation and execution
        fetch('/api/make_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                from_position: fromPosition,
                to_position: toPosition
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showModal(data.error);
                renderBoard();
                return;
            }
            
            // Record that we used this die
            if (!gameState.usedDice) {
                gameState.usedDice = [];
            }
            gameState.usedDice.push(dieValue);
            console.log(`Added die value ${dieValue} to used dice. Current used dice:`, gameState.usedDice);
            
            // Update client-side move validator state for future validations
            if (data.board) {
                if (fromPosition === 23) {
                    // NUCLEAR OPTION: Set this flag aggressively and permanently block position 23
                    gameState.headMoveMade = true;
                    gameState.headMovesMade += 1;
                    console.log(`🚫 NUCLEAR TRACKING: Updated head move tracking: headMoveMade=${gameState.headMoveMade}, headMovesMade=${gameState.headMovesMade}`);
                    
                    // Remove head position from valid moves immediately
                    console.log("🚫 NUCLEAR BLOCKING: Removing position 23 from valid moves");
                    if (gameState.validMovesByPiece && gameState.validMovesByPiece[23]) {
                        delete gameState.validMovesByPiece[23];
                    }
                    
                    // Add global variable to forcefully block position 23
                    window.headMoveBlockActive = true;
                }
            }
            
            // Update board
            if (data.board) {
                gameState.board = data.board;
            }
            
            // Update borne off counts
            if (data.borne_off) {
                gameState.borneOff = data.borne_off;
                renderBorneOff();
            }
            
            // Check if game is over
            if (data.game_over) {
                const winner = data.winner === 'white' ? 'You (White)' : 'AI (Black)';
                showModal(`Game Over! ${winner} wins!`);
                updateGameStatus(`Game Over! ${winner} wins!`);
                renderBoard();
                return;
            }
            
            // Check if this was the first move of a two-move sequence
            if (data.first_move_complete && data.needs_second_move) {
                gameState.firstMoveMade = true;
                
                // Update client-side validator state
                // Calculate the die value used in this move
                const dieValue = moveValidator.getMoveDistance(fromPosition, toPosition);
                moveValidator.updateDiceAfterMove(dieValue);
                
                // Update valid moves for second move if provided
                if (data.valid_moves_by_piece) {
                    console.log("Updating valid moves for second move:", data.valid_moves_by_piece);
                    gameState.validMovesByPiece = data.valid_moves_by_piece;
                    
                    // FORCE head move removal regardless of count - this is a more aggressive approach
                    if (fromPosition === gameState.HEAD_POSITIONS?.white) {
                        console.log("HEAD MOVE FORCED FILTERING: Removing head position 23 from valid moves after head move");
                        delete gameState.validMovesByPiece[gameState.HEAD_POSITIONS.white];
                    }
                    
                    // Normal filtering based on count
                    if (gameState.headMovesMade >= 1) {
                        // Special case for first turn doubles
                        const isFirstTurnSpecialDoubles = 
                            gameState.isFirstTurn && 
                            gameState.dice && 
                            gameState.dice.length >= 2 && 
                            gameState.dice[0] === gameState.dice[1] && 
                            [3, 4, 6].includes(gameState.dice[0]);
                            
                        // If we've already used our allowed head moves, filter out position 23
                        if (gameState.headMovesMade >= (isFirstTurnSpecialDoubles ? 2 : 1)) {
                            // Remove head position from valid moves
                            console.log("Filtering out head position from valid moves after first move");
                            delete gameState.validMovesByPiece[gameState.HEAD_POSITIONS.white];
                        }
                    }
                    
                    // Log to server
                    fetch('/api/log', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: `Client: Received updated valid moves for second move: ${JSON.stringify(data.valid_moves_by_piece)}`
                        })
                    });
                } else {
                    // If server didn't provide updated moves, calculate them client-side
                    gameState.validMovesByPiece = moveValidator.getAllValidMoves();
                    
                    // FORCE head move removal
                    if (fromPosition === gameState.HEAD_POSITIONS?.white) {
                        console.log("HEAD MOVE FORCED FILTERING: Removing head position 23 from client-calculated valid moves after head move");
                        delete gameState.validMovesByPiece[gameState.HEAD_POSITIONS.white];
                    }
                    
                    // Same filtering for client-side calculated moves
                    if (gameState.headMovesMade >= 1) {
                        const isFirstTurnSpecialDoubles = 
                            gameState.isFirstTurn && 
                            gameState.dice && 
                            gameState.dice.length >= 2 && 
                            gameState.dice[0] === gameState.dice[1] && 
                            [3, 4, 6].includes(gameState.dice[0]);
                            
                        if (gameState.headMovesMade >= (isFirstTurnSpecialDoubles ? 2 : 1)) {
                            console.log("Filtering out head position from client-calculated valid moves");
                            delete gameState.validMovesByPiece[gameState.HEAD_POSITIONS.white];
                        }
                    }
                }
                
                renderBoard();
                updateGameStatus('Select your second move');
                return;
            }
            
            // If the move is complete, update for AI's turn
            if (data.current_player === 'white') {
                // Back to human's turn
                gameState.currentPlayer = 'white';
                gameState.dice = data.dice;
                gameState.validMovesByPiece = data.valid_moves_by_piece || {};
                gameState.firstMoveMade = false;
                
                // Only reset head move tracking if AI has actually moved
                // This ensures we don't reset during the player's multi-move turn
                if (data.ai_moves && data.ai_moves.length > 0) {
                    console.log("AI has moved, resetting head move tracking for new turn");
                    gameState.headMoveMade = false; 
                    gameState.headMovesMade = 0;
                    // Also reset global block if it was set
                    window.headMoveBlockActive = false;
                } else {
                    console.log("Keeping head move tracking intact: headMoveMade=" + 
                                gameState.headMoveMade + ", headMovesMade=" + gameState.headMovesMade);
                }
                
                gameState.isFirstTurn = data.first_turn || false;
                gameState.usedDice = []; // Reset used dice for new turn
                
                // Initialize move validator with new dice
                console.log("Initializing move validator with reset head move state for a new turn");
                moveValidator.initialize(
                    gameState.board,
                    gameState.dice,
                    gameState.currentPlayer,
                    gameState.isFirstTurn,
                    false // headMoveMade reset for new turn
                );
                
                renderDice();
                renderBoard();
                
                // Show AI's moves
                if (data.ai_moves) {
                    let aiMoveText = 'AI moved: ';
                    data.ai_moves.forEach((move, index) => {
                        const from = move.from;
                        const to = move.to === -1 ? 'off' : move.to;
                        aiMoveText += `${from} → ${to}`;
                        if (index < data.ai_moves.length - 1) {
                            aiMoveText += ', ';
                        }
                    });
                    
                    let aiDiceText = data.ai_dice ? ' [AI Roll: ' + data.ai_dice.join(', ') + ']' : '';
                    
                    // Check if doubles - note we might have more than 2 moves with doubles
                    let isDoubles = data.ai_dice && data.ai_dice[0] === data.ai_dice[1];
                    let movesCount = data.ai_moves ? data.ai_moves.length : 0;
                    
                    if (data.ai_had_no_moves) {
                        updateGameStatus('AI had no valid moves' + aiDiceText + '. Your turn - Roll: ' + gameState.dice.join(', '));
                    } else if (data.ai_moved_from_head && movesCount === 1) {
                        updateGameStatus(aiMoveText + ' (only one move allowed from head position)' + aiDiceText + '. Your turn - Roll: ' + gameState.dice.join(', '));
                    } else if (isDoubles && movesCount < 4) {
                        updateGameStatus(aiMoveText + ' (using ' + movesCount + ' of 4 possible moves)' + aiDiceText + '. Your turn - Roll: ' + gameState.dice.join(', '));
                    } else {
                        updateGameStatus(aiMoveText + aiDiceText + '. Your turn - Roll: ' + gameState.dice.join(', '));
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error making move:', error);
            renderBoard();
        });
    }

    // Update game status
    function updateGameStatus(message) {
        gameStatus.textContent = message;
    }

    // Show modal
    function showModal(message) {
        modalMessage.textContent = message;
        modal.style.display = 'block';
    }

    // Initialize the board
    function init() {
        updateGameStatus('Press "New Game" to start');
    }

    // Initialize the game
    init();
});
