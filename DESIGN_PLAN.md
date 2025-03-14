# Long Nardy Web Interface Implementation Plan

## Overview
We need to implement a web interface for Long Nardy by adapting the backgammon.js UI while maintaining our Python backend. The goal is to make our UI match backgammon.js as closely as possible, with changes only where necessary to support Long Nardy's specific rules.

## UI Implementation Approach
We will closely replicate the backgammon.js UI with these principles:

1. **Direct Visual Adaptation**:
   - Use the same board layout, styling, and components
   - Maintain the same UI organization (game menu, dice display, action buttons)
   - Keep identical animations and visual effects
   - Preserve the responsive design approach

2. **Required Rule-Based UI Modifications**:
   - Adapt for Long Nardy starting positions (White's 15 checkers on point 24; Black's 15 on point 12)
   - Modify for CCW movement direction for both players
   - Home areas visualization (White 1–6, Black 13–18)
   - Visual indicators for head positions (White 24, Black 12)
   - Adapt bearing off visualization
   - Add indicators for the Block Rule when relevant (contiguous 6 checkers)

3. **Component Structure**:
   - Implement `SimpleBoardUI` as the main class, matching backgammon.js structure
   - Recreate all UI elements: fields, points, pieces, dice panels, action buttons
   - Use the same CSS class names and styling where possible
   - Replicate exact positioning and stacking logic for pieces

## Frontend Implementation Details

### HTML Structure
We will closely match the backgammon.js HTML structure:

```html
<!-- Core Board Template -->
<div id="tmpl-board">
  <div id="frame-top" class="frame"></div>
  <div class="board cf">
    <div id="pane-left" class="pane">
      <div id="field0" class="field row0 col0"></div>
      <div id="field1" class="field row1 col0"></div>
      <div class="dice-panel left">
        <div id="dice-left" class="dice left"></div>
      </div>
    </div>
    <div id="bar" class="bar">
      <table width="100%" height="100%">
        <tr height="10%">
          <td valign="top">[Menu Button]</td>
        </tr>
        <tr height="45%">
          <td valign="top" id="top-bar"></td>
        </tr>
        <tr height="45%">
          <td valign="bottom" id="bottom-bar"></td>
        </tr>
      </table>
    </div>
    <div id="pane-right" class="pane">
      <div id="field2" class="field row0 col1"></div>
      <div id="field3" class="field row1 col1"></div>
      <div class="dice-panel right">
        <div id="dice-right" class="dice right"></div>
      </div>
    </div>
  </div>
  <div id="frame-bottom" class="frame"></div>
  <div class="action-panel">
    <button id="btn-roll" class="btn btn-primary btn-lg action">Roll</button>
    <button id="btn-confirm" class="btn btn-primary btn-lg action">Confirm</button>
    <button id="btn-undo" class="btn btn-default btn-lg action">Undo</button>
  </div>
</div>
```

### CSS Structure
We'll implement the same CSS structure as backgammon.js:

1. **Base Styles**: Reset, layout, containers
2. **Board Elements**: Board, fields, points, bar
3. **Dice Styling**: Dice panels, die faces, animations
4. **Pieces**: Piece colors, stacking behavior, hover/active states
5. **Game Info**: Player info, scores, game status
6. **Action Controls**: Buttons, panels
7. **Responsive Rules**: Media queries for different devices

### JavaScript Components

We'll implement the following main components matching backgammon.js:

1. **SimpleBoardUI** (Main component):
   ```javascript
   function SimpleBoardUI(client) {
     this.client = client;
     this.match = null;
     this.rule = null;
     this.fields = [];
     // Initialize board elements
     this.init();
   }
   
   SimpleBoardUI.prototype = {
     // Core methods
     init: function() { /* initialize board */ },
     handlePointClick: function(e) { /* handle point clicks */ },
     createPoints: function() { /* create point elements */ },
     createPieces: function() { /* create pieces based on game state */ },
     compactPosition: function(pos) { /* stack pieces at a position */ },
     updateBoard: function() { /* update UI based on game state */ },
     updateDice: function(dice, type) { /* update dice display */ },
     // Additional methods for move handling, bearoff, etc.
   };
   ```

2. **App/Client** (Communication & Game Logic):
   ```javascript
   function App() {
     this.config = { /* configuration */ };
     this.player = null;
     this.ui = null;
     this.init();
   }
   
   App.prototype = {
     init: function() { /* initialize app and connect UI */ },
     reqRollDice: function() { /* request dice roll */ },
     reqMove: function(piece, steps) { /* request piece move */ },
     reqConfirmMoves: function() { /* confirm turn end */ },
     reqUndoMoves: function() { /* undo moves */ },
     updateUI: function() { /* update UI after state change */ }
   };
   ```

3. **Communication** (Event Handling):
   ```javascript
   function Comm() {
     this.subscribers = {};
     this.init();
   }
   
   Comm.prototype = {
     init: function() { /* setup event listeners */ },
     subscribe: function(eventName, callback) { /* register event handler */ },
     dispatch: function(eventName, payload) { /* trigger event */ },
     send: function(msg, payload) { /* send message to server */ }
   };
   ```

### Visual Assets and Animations

1. **Dice**: 
   - Six unique dice faces with proper styling
   - Random rotation angles for realistic feel
   - "Played" state with opacity change

2. **Pieces**:
   - White and black checkers with consistent styling
   - Stacking with proper overlaps
   - Hover and selection effects

3. **Animations**:
   - Dice rolling animation
   - Piece movement with smooth transitions
   - Notification slide-ins and fades

## Backend Integration

- Keep our Python/Flask backend for game logic
- Ensure the API provides all necessary data for the UI to function
- Modify backend responses to match what the UI expects
- Make sure game state information is complete

## Communication Protocol

After analyzing backgammon.js, we can see it uses a socket-based communication system with structured message passing. Here's how we'll approach our protocol:

### Backgammon.js Protocol Analysis

The backgammon.js system uses:
- WebSockets for real-time communication
- Message objects with ID, sequence number, and parameters
- Bidirectional event-based communication (client ↔ server)
- Subscription model for handling various game events
- Message types for different game actions (roll dice, move piece, confirm moves, etc.)

Key message types include:
- `EVENT_MATCH_START` - Signals the start of a match
- `EVENT_MATCH_OVER` - Signals the end of a match
- `EVENT_PLAYER_JOINED` - Notifies when a player joins a match
- `JOIN_MATCH` - Request to join an existing match
- `CREATE_MATCH` - Request to create a new match
- Roll dice, move piece, confirm moves, undo moves, resign, etc.

### Our Communication Approach

Since we're using a RESTful API with our Python backend instead of WebSockets, we'll implement a hybrid approach:

#### 1. Enhanced RESTful API Endpoints

| Endpoint | Method | Purpose | Status | Parameters | Response |
|----------|--------|---------|--------|------------|----------|
| `/api/new_game` | POST | Start a new game | Exists | None | Game state with board, dice, valid moves |
| `/api/roll_dice` | POST | Roll dice for current player | New | None | New dice values, valid moves |
| `/api/get_valid_moves` | POST | Get valid moves for a piece | Exists | `position` | List of valid destinations |
| `/api/make_move` | POST | Execute a single move | Exists | `from_position`, `to_position` | Updated game state |
| `/api/confirm_moves` | POST | Confirm end of turn | New | None | AI turn results, new game state |
| `/api/undo_moves` | POST | Undo moves in current turn | New | None | Previous game state |
| `/api/game_state` | GET | Get current game state | New | None | Complete game state |
| `/api/resign` | POST | Resign from game | New | None | Game over message |

#### 2. Client-Side Event System

We'll implement a message bus in JavaScript that mirrors backgammon.js event handling but works with our REST API:

```javascript
// Message types matching backgammon.js where possible
const Message = {
  // Game events
  EVENT_MATCH_START: 'match_start',
  EVENT_MATCH_OVER: 'match_over',
  EVENT_TURN_START: 'turn_start',
  EVENT_DICE_ROLLED: 'dice_rolled',
  EVENT_MOVE_EXECUTED: 'move_executed',
  EVENT_MOVES_CONFIRMED: 'moves_confirmed',
  EVENT_MOVES_UNDONE: 'moves_undone',
  
  // Requests
  REQ_ROLL_DICE: 'roll_dice',
  REQ_MOVE: 'move',
  REQ_CONFIRM: 'confirm',
  REQ_UNDO: 'undo',
  REQ_RESIGN: 'resign'
};
```

#### 3. Message Payload Structure

We'll structure our API payloads to match what the UI expects from backgammon.js:

**Example for make_move response:**
```json
{
  "message": "move_executed",
  "game": {
    "board": [15, 0, 0, ...],
    "turnPlayer": "white",
    "turnDice": [4, 2],
    "isOver": false
  },
  "move": {
    "from": 23,
    "to": 19,
    "piece": {
      "id": 12,
      "type": "white"
    }
  },
  "hasMoreMoves": true,
  "validMoves": [
    {"from": 19, "to": 17}, 
    {"from": 19, "to": 15}
  ]
}
```

#### 4. API Wrapper Layer

We'll create a wrapper class that translates REST calls to match backgammon.js event system:

```javascript
class ApiWrapper {
  constructor(comm) {
    this.comm = comm;
  }
  
  // Maps backgammon.js reqRollDice() to our API
  reqRollDice() {
    fetch('/api/roll_dice', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
      // Dispatch event as if it came from a WebSocket
      this.comm.dispatch(Message.EVENT_DICE_ROLLED, {
        dice: data.dice,
        validMoves: data.valid_moves
      });
    });
  }
  
  // Similar methods for other API calls
}
```

This approach allows the UI components to work with the same event structure they expect from backgammon.js, while our backend continues to use a RESTful API.

## Long Nardy Rule Implementation

Long Nardy rules to be visually represented:

1. **Setup**: Show White's 15 checkers on point 24; Black's 15 on point 12.

2. **Movement**: Visualize CCW movement for both players and highlight home areas (White 1–6, Black 13–18).

3. **Head Rule**: Visual indicators when only 1 checker may leave the head position per turn, with special indication for the first turn exception (doubles 6, 4, or 3).

4. **Block Rule**: Provide visual warning when approaching a contiguous block of 6 checkers that would trap opponent pieces.

5. **Bearing Off**: Clearly indicate when bearing off is possible and which pieces are eligible.

## Implementation Sequence

### Phase 1: Foundation (Days 1-2) ✅ COMPLETED
1. ✅ Set up project structure mirroring backgammon.js organization
2. ✅ Implement base HTML templates with board layout
3. ✅ Port CSS styling from backgammon.js
4. ✅ Create basic JavaScript component skeletons
5. ✅ Implement event system based on backgammon.js model

### Phase 2: Core UI (Days 3-4) ✅ COMPLETED
6. ✅ Implement board rendering with points and fields
7. ✅ Create piece visualization and positioning
8. ✅ Implement dice appearance and action buttons
9. ✅ Add game status displays and overlays
10. ✅ Connect the components through the event system

### Phase 3: Game Mechanics (Days 5-6) ✅ COMPLETED
11. ✅ Implement piece movement with drag and drop
12. ✅ Create move validation UI feedback
13. ✅ Implement dice rolling and move selection
14. ✅ Add piece stacking and positioning logic
15. ✅ Create turn management and confirmation flow

### Phase 4: Long Nardy Rules (Day 7) ✅ COMPLETED
16. ✅ Implement head rule visualization and enforcement
17. ✅ Add block rule warnings and validation
18. ✅ Create bearing off visualization and validation
19. ✅ Adjust for Long Nardy-specific board layout
20. ✅ Implement special first-turn rules for doubles

### Phase 5: Backend Integration (Days 8-9) ✅ COMPLETED
21. ✅ Modify backend API to match UI expectations
22. ✅ Implement client-side API communication
23. ✅ Create state synchronization between frontend and backend
24. ✅ Add error handling and move validation feedback
25. ✅ Implement game flow from start to completion

### Phase 6: Polish and Testing (Day 10) ✅ COMPLETED
26. ✅ Add animations and transitions
27. ✅ Implement responsive design adaptations
28. ✅ Improved visual appearance with enhanced assets
29. ✅ Tested on different devices and browsers
30. ✅ Applied final bug fixes and visual polish

## Project Organization

We'll organize the web interface following a structure similar to backgammon.js:

```
/web
  /static
    /css
      backgammon.css     # Main CSS file (renamed from style.css)
      ribbons.css        # Additional styling
    /js
      /components
        SimpleBoardUI.js  # Main board component
        App.js            # Application logic
        Comm.js           # Communication layer
        ApiWrapper.js     # REST API adapter
      /lib
        model.js          # Game model/data structures
      /utils
        helpers.js        # Utility functions
      main.js             # Entry point
    /images
      # Dice, pieces, board elements
  /templates
    index.html           # Main page
  /server
    app.py               # Flask application
    narde_patched.py     # Game logic integration
```

## Success Criteria

- UI is visually identical to backgammon.js except where Long Nardy rules require differences
- All Long Nardy rules are properly visualized
- Game flow matches the expected experience
- Integration with our Python backend is seamless
- Performance is optimized for smooth gameplay
- Code organization follows similar patterns to backgammon.js
- Mobile and desktop compatibility is maintained