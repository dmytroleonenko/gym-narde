#!/usr/bin/env python3
import pygame
import sys
import numpy as np
from gym_narde.envs.narde_env import NardeEnv
from gym_narde.envs.narde import rotate_board

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)
LIGHT_BROWN = (205, 133, 63)
DARK_BROWN = (93, 64, 55)
RED = (255, 0, 0)
GREEN = (76, 175, 80)
BLUE = (30, 144, 255)
GRAY = (169, 169, 169)
GOLD = (255, 215, 0)  # Gold for selected points
LIGHT_GREEN = (173, 255, 47)  # GreenYellow for movable points
HIGHLIGHT = (255, 193, 7, 180)  # Semi-transparent yellow

# Screen dimensions
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
POINT_WIDTH = 60
POINT_HEIGHT = 250
CENTER_BAR_WIDTH = 50
CHECKER_SIZE = 45
DICE_SIZE = 80
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 50
POINT_TRIANGLE_HEIGHT = 220

# Game constants
BOARD_POSITIONS = 24

class LongNardeGame:
    def __init__(self):
        # Create the Narde environment
        self.env = NardeEnv()
        self.env.reset()
        # Initialize dice for environment to prevent AttributeError
        self.env.dice = [0, 0]
        
        # Initialize pygame screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Long Narde Manual Play")
        self.font = pygame.font.SysFont('Arial', 20)
        self.small_font = pygame.font.SysFont('Arial', 16)
        self.large_font = pygame.font.SysFont('Arial', 24, bold=True)
        
        # Game state variables
        self.selected_point = None
        self.dice = [1, 1]  # Initialize with dummy dice
        self.dice_used = [False] * 4  # Track up to 4 moves for doubles
        self.current_player = 1  # Start with White player (1)
        self.valid_moves = []
        self.valid_destination_points = []
        self.roll_entered = False
        self.game_over = False
        self.message = "Enter dice values (1-6) and click Roll"
        self.is_doubles = False
        
        # Dragging state
        self.dragging = False
        self.drag_checker = None
        self.drag_source = None
        self.drag_pos = (0, 0)
        
        # Track last move for visualization
        self.last_move_from = None
        self.last_move_to = None
        
        # For manual dice entry
        self.dice_input = ["", ""]
        self.active_dice_input = 0
        
        # Create point coordinates lookup based on current player's perspective
        self.point_coordinates = self.create_point_coordinates()
        
        # Log the initial board state
        self.log_board_state("Initial board state")
    
    def log_board_state(self, message):
        """Log the current board state to console"""
        # Get the actual board state (not perspective board)
        board = self.env.game.board
        
        print(f"\n{message}")
        print("Board state (from white player's view):")
        print("Position:  ", end="")
        for i in range(24):
            print(f"{i+1:3d}", end=" ")
        print("\nCheckers:  ", end="")
        for i in range(24):
            print(f"{board[i]:3d}", end=" ")
        print("\n")
        
        print(f"Current player: {self.current_player_name()}")
        print(f"Dice: {self.dice}, Used: {self.dice_used}")
        print("Valid moves:", self.valid_moves)
        print("=" * 80)
    
    def create_point_coordinates(self):
        """Create and store point coordinates for the Narde board"""
        # Calculate board position
        board_left = (SCREEN_WIDTH - (12 * POINT_WIDTH + CENTER_BAR_WIDTH)) / 2
        board_top = (SCREEN_HEIGHT - POINT_HEIGHT * 2) / 2
        
        # Store point coordinates in a dictionary (pointIdx -> (x, y, is_top_row))
        coordinates = {}
        
        # Bottom row points (24-13) from left to right
        for i in range(12):
            x = board_left + i * POINT_WIDTH
            if i >= 6:
                x += CENTER_BAR_WIDTH
            coordinates[23 - i] = (x, board_top + POINT_HEIGHT, False)  # False indicates bottom row
        
        # Top row points (1-12) from left to right
        for i in range(12):
            x = board_left + i * POINT_WIDTH
            if i >= 6:
                x += CENTER_BAR_WIDTH
            coordinates[i] = (x, board_top, True)  # True indicates top row
        
        return coordinates
    
    def roll_dice(self):
        """Set up dice based on user input"""
        try:
            die1 = int(self.dice_input[0]) if self.dice_input[0] else 1
            die2 = int(self.dice_input[1]) if self.dice_input[1] else 1
            
            # Validate dice values
            if not (1 <= die1 <= 6 and 1 <= die2 <= 6):
                self.message = "Dice must be between 1 and 6"
                return False
            
            # Sort dice for proper move validation (higher die first)
            self.dice = sorted([die1, die2], reverse=True)
            self.is_doubles = die1 == die2
            self.dice_used = [False] * (4 if self.is_doubles else 2)
            self.roll_entered = True
            
            # Get valid moves for current player
            self.update_valid_moves()
            
            # Special message for first turn doubles
            is_first_turn = (self.current_player == 1 and self.env.game.first_turn_white) or \
                          (self.current_player == -1 and self.env.game.first_turn_black)
            if is_first_turn and self.is_doubles and die1 in [3, 4, 6]:
                self.message = f"First turn doubles {die1}! You can move four times, including two from head."
            elif self.is_doubles:
                self.message = f"Doubles {die1}! You can move four times."
            
            if len(self.valid_moves) == 0:
                self.message = f"No valid moves for {self.current_player_name()}. Skip turn."
                self.dice_used = [True] * len(self.dice_used)  # Mark all dice as used
            else:
                if not self.message.startswith("First turn") and not self.message.startswith("Doubles"):
                    self.message = f"{self.current_player_name()}'s turn. Valid moves: {len(self.valid_moves)}"
            
            # Log the roll and valid moves
            self.log_board_state(f"{self.current_player_name()} rolled {die1}, {die2}")
            
            return True
            
        except ValueError:
            self.message = "Please enter valid numbers for dice"
            return False
    
    def update_valid_moves(self):
        """Update valid moves based on current dice and board state"""
        # Update dice in the environment for observation generation
        self.env.dice = self.dice.copy()
        
        # Get remaining dice values based on unused dice
        remaining_dice = []
        if self.is_doubles:
            # For doubles, each unused die adds one value
            remaining_dice = [self.dice[0] for _ in range(self.dice_used.count(False))]
        else:
            # For non-doubles, just get the unused dice
            for i, used in enumerate(self.dice_used):
                if not used:
                    remaining_dice.append(self.dice[i])
        
        # Get valid moves from the game - always treat as White's turn
        if remaining_dice:
            self.valid_moves = self.env.game.get_valid_moves(remaining_dice, 1)  # Always use 1 for White
        else:
            self.valid_moves = []
            
        # Update valid destination points
        self.valid_destination_points = []
        
        # If a point is selected, filter valid moves for that source
        if self.selected_point is not None:
            self.valid_destination_points = [move[1] for move in self.valid_moves if move[0] == self.selected_point]
    
    def select_point(self, point_idx):
        """Select a point on the board"""
        # Can only select if dice have been rolled
        if not self.roll_entered or all(self.dice_used):
            return
        
        # Check if this is a valid source point
        valid_sources = list(set(move[0] for move in self.valid_moves))
        
        if point_idx in valid_sources:
            self.selected_point = point_idx
            # Update valid destination points for this source
            self.valid_destination_points = [move[1] for move in self.valid_moves if move[0] == point_idx]
        else:
            self.selected_point = None
            self.valid_destination_points = []
    
    def move_checker(self, to_point):
        """Move a checker from selected point to destination"""
        # Can only move if a point is selected
        if self.selected_point is None:
            return False
        
        # Check if this is a valid move
        move = (self.selected_point, to_point)
        if to_point == 'off':
            valid_moves_with_off = [m for m in self.valid_moves if m[0] == self.selected_point and m[1] == 'off']
            if valid_moves_with_off:
                move = valid_moves_with_off[0]
            else:
                return False
        elif move not in self.valid_moves:
            return False
        
        # Find which die was used for this move
        used_die_idx = self.find_used_die(move)
        if used_die_idx is None:
            return False
        
        # Execute the move - always as if we're White (player 1)
        self.env.game._execute_move(move, 1)
        
        # Mark the die as used
        self.dice_used[used_die_idx] = True
        
        # Track last move for visualization
        self.last_move_from = self.selected_point
        self.last_move_to = to_point
        
        # Log the move
        log_msg = f"Move: {self.current_player_name()} moved from {self.selected_point+1} to "
        log_msg += "bearing off" if to_point == 'off' else f"{to_point+1}"
        self.log_board_state(log_msg)
        
        # Reset selection
        self.selected_point = None
        self.valid_destination_points = []
        
        # Update valid moves based on remaining dice
        self.update_valid_moves()
        
        # Check if turn is over
        if all(self.dice_used) or len(self.valid_moves) == 0:
            self.end_turn()
        
        return True
    
    def find_used_die(self, move):
        """Find which die was used for a move"""
        from_pos, to_pos = move
        
        # Get the actual distance for the move
        if to_pos == 'off':
            # For bearing off, the distance is from the point to position 0
            distance = from_pos + 1
        else:
            distance = from_pos - to_pos
        
        # For doubles, find the first unused die (they're all the same value)
        if self.is_doubles:
            for i in range(len(self.dice_used)):
                if not self.dice_used[i] and self.dice[0] == distance:
                    return i
        else:
            # For non-doubles, find matching unused die
            for i, (die, used) in enumerate(zip(self.dice, self.dice_used)):
                if not used and distance == die:
                    return i
        
        # If no exact match and this is a bearing off move, check for higher die
        if to_pos == 'off':
            # Check if all checkers are in home (required for bearing off)
            home_range = range(6) if self.current_player == 1 else range(18, 24)
            all_in_home = True
            board = self.env.game.get_perspective_board(self.current_player)
            for i in range(24):
                if i not in home_range and board[i] * self.current_player > 0:
                    all_in_home = False
                    break
            
            if all_in_home:
                # Can use a higher die if no checkers are further from home
                for i, (die, used) in enumerate(zip(self.dice, self.dice_used)):
                    if not used and die > distance:
                        # Check no checkers are further from home
                        if not any(board[j] * self.current_player > 0 for j in range(from_pos)):
                            return i
        
        return None
    
    def end_turn(self):
        """End the current player's turn"""
        # Switch player
        self.current_player *= -1
        
        # Reset dice state
        self.dice = [1, 1]
        self.dice_used = [False] * 4  # Reset to maximum size for doubles
        self.roll_entered = False
        self.dice_input = ["", ""]
        self.is_doubles = False
        
        # Reset selection
        self.selected_point = None
        self.valid_destination_points = []
        
        # Rotate the board in the environment and swap borne off counters
        self.env.game.board = rotate_board(self.env.game.board)
        self.env.game.first_turn_white, self.env.game.first_turn_black = self.env.game.first_turn_black, self.env.game.first_turn_white
        self.env.game.borne_off_white, self.env.game.borne_off_black = self.env.game.borne_off_black, self.env.game.borne_off_white
        
        # Update message
        self.message = f"{self.current_player_name()}'s turn. Roll dice."
        
        # Check for game over
        if self.env.game.borne_off_white >= 15 or self.env.game.borne_off_black >= 15:
            self.game_over = True
            winner = "White" if self.env.game.borne_off_white >= 15 else "Black"
            self.message = f"Game Over! {winner} wins!"
            self.log_board_state(f"Game Over! {winner} wins!")
        else:
            # Log the board state from new player's perspective
            print(f"\nBoard state (from white player's view):")
            print("Position:  ", end="")
            for i in range(24):
                print(f"{i+1:3d}", end=" ")
            print("\nCheckers:  ", end="")
            board = self.env.game.board
            for i in range(24):
                print(f"{board[i]:3d}", end=" ")
            print("\n")
            print(f"Current player: {self.current_player_name()}")
            print("=" * 80)
    
    def current_player_name(self):
        """Get the name of the current player"""
        return "White" if self.current_player == 1 else "Black"
    
    def handle_event(self, event):
        """Handle pygame events"""
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Handle mouse button down (start of drag or click)
            mouse_pos = pygame.mouse.get_pos()
            
            # Check if point on board was clicked
            point_idx = self.get_point_at_pos(mouse_pos)
            
            # Start dragging a checker
            if point_idx is not None and self.roll_entered and not all(self.dice_used):
                valid_sources = list(set(move[0] for move in self.valid_moves))
                if point_idx in valid_sources:
                    self.drag_source = point_idx
                    self.dragging = True
                    self.drag_pos = mouse_pos
                    # Don't select the point immediately, wait for mouse up or drag
            
            # Check if dice input areas were clicked
            for i in range(2):
                if self.is_dice_input_clicked(mouse_pos, i):
                    self.active_dice_input = i
            
            # Check if roll button was clicked
            if self.is_roll_button_clicked(mouse_pos) and not self.roll_entered:
                self.roll_dice()
            
            # Check if skip button was clicked
            if self.is_skip_button_clicked(mouse_pos) and self.roll_entered:
                self.end_turn()
        
        elif event.type == pygame.MOUSEMOTION:
            # Handle mouse motion for dragging
            if self.dragging:
                self.drag_pos = event.pos
        
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            # Handle mouse button up (end of drag or click)
            if self.dragging:
                # Finish the drag operation
                mouse_pos = event.pos
                destination_idx = self.get_point_at_pos(mouse_pos)
                
                # If we have a valid source and destination
                if self.drag_source is not None:
                    self.selected_point = self.drag_source
                    
                    # Update valid destinations
                    self.valid_destination_points = [move[1] for move in self.valid_moves 
                                                  if move[0] == self.drag_source]
                    
                    # Dropped on a valid destination
                    if destination_idx is not None and destination_idx in self.valid_destination_points:
                        self.move_checker(destination_idx)
                    # Dropped on bearing off area
                    elif self.can_bear_off() and self.is_bearing_off_area_clicked(mouse_pos):
                        self.move_checker('off')
                
                # Reset drag state
                self.dragging = False
                self.drag_source = None
            else:
                # Normal click (no drag involved)
                mouse_pos = event.pos
                
                # Check if point on board was clicked
                point_idx = self.get_point_at_pos(mouse_pos)
                if point_idx is not None:
                    # If clicking a valid source point, select it
                    valid_sources = list(set(move[0] for move in self.valid_moves))
                    if point_idx in valid_sources:
                        self.select_point(point_idx)
                    # If clicking a valid destination, move there
                    elif self.selected_point is not None and point_idx in self.valid_destination_points:
                        self.move_checker(point_idx)
                
                # Check if bearing off area was clicked
                if self.can_bear_off() and self.is_bearing_off_area_clicked(mouse_pos):
                    self.move_checker('off')
        
        elif event.type == pygame.KEYDOWN:
            # Handle keyboard input for dice
            if not self.roll_entered:
                if event.key == pygame.K_TAB:
                    self.active_dice_input = (self.active_dice_input + 1) % 2
                elif event.key == pygame.K_RETURN:
                    self.roll_dice()
                elif event.key == pygame.K_BACKSPACE:
                    self.dice_input[self.active_dice_input] = self.dice_input[self.active_dice_input][:-1]
                elif event.unicode.isdigit() and len(self.dice_input[self.active_dice_input]) < 1:
                    self.dice_input[self.active_dice_input] += event.unicode
    
    def get_point_at_pos(self, pos):
        """Get the point index at mouse position"""
        mouse_x, mouse_y = pos
        
        # Calculate board position
        board_left = (SCREEN_WIDTH - (12 * POINT_WIDTH + CENTER_BAR_WIDTH)) / 2
        board_top = (SCREEN_HEIGHT - POINT_HEIGHT * 2) / 2
        
        # Check each point's area to see if it contains the mouse position
        for point_idx, (x, y, is_top_row) in self.point_coordinates.items():
            # Point area
            rect_width = POINT_WIDTH
            rect_height = POINT_HEIGHT
            
            # Check if mouse is in the point's rectangle
            if (x <= mouse_x <= x + rect_width and 
                y <= mouse_y <= y + rect_height):
                # For triangular points, check if mouse is in the triangle area
                point_y = y
                if is_top_row:
                    # Top row triangles point down
                    triangle_height = POINT_TRIANGLE_HEIGHT
                    # Check if in triangular area (y coordinate is within triangle height)
                    if mouse_y <= point_y + triangle_height:
                        # Check if within the triangle x bounds
                        # Base of triangle is at top, narrows toward bottom
                        progress = (mouse_y - point_y) / triangle_height
                        triangle_width = POINT_WIDTH * (1 - progress)
                        triangle_left = x + (POINT_WIDTH - triangle_width) / 2
                        if triangle_left <= mouse_x <= triangle_left + triangle_width:
                            return point_idx
                else:
                    # Bottom row triangles point up
                    triangle_height = POINT_TRIANGLE_HEIGHT
                    # Check if in triangular area (y coordinate is within triangle height)
                    if mouse_y >= point_y + (POINT_HEIGHT - triangle_height):
                        # Check if within the triangle x bounds
                        # Base of triangle is at bottom, narrows toward top
                        progress = (point_y + POINT_HEIGHT - mouse_y) / triangle_height
                        triangle_width = POINT_WIDTH * (1 - progress)
                        triangle_left = x + (POINT_WIDTH - triangle_width) / 2
                        if triangle_left <= mouse_x <= triangle_left + triangle_width:
                            return point_idx
        
        return None
    
    def is_bearing_off_area_clicked(self, pos):
        """Check if the bearing off area was clicked"""
        mouse_x, mouse_y = pos
        
        # Calculate board position
        board_left = (SCREEN_WIDTH - (12 * POINT_WIDTH + CENTER_BAR_WIDTH)) / 2
        board_top = (SCREEN_HEIGHT - POINT_HEIGHT * 2) / 2
        
        # Bearing off area on the right side of the board
        # For White player, bearing off area should be in the bottom right
        bearing_off_x = board_left + CENTER_BAR_WIDTH + 12 * POINT_WIDTH
        bearing_off_y = board_top
        
        return (bearing_off_x <= mouse_x <= bearing_off_x + POINT_WIDTH and
                bearing_off_y <= mouse_y <= bearing_off_y + POINT_HEIGHT * 2)
    
    def is_dice_input_clicked(self, pos, dice_idx):
        """Check if a dice input area was clicked"""
        mouse_x, mouse_y = pos
        
        dice_x = SCREEN_WIDTH / 2 - 100 + dice_idx * 120
        dice_y = SCREEN_HEIGHT - 120
        
        return (dice_x <= mouse_x <= dice_x + 80 and
                dice_y <= mouse_y <= dice_y + 40)
    
    def is_roll_button_clicked(self, pos):
        """Check if the roll button was clicked"""
        mouse_x, mouse_y = pos
        
        button_x = SCREEN_WIDTH / 2 - BUTTON_WIDTH - 20
        button_y = SCREEN_HEIGHT - 60
        
        return (button_x <= mouse_x <= button_x + BUTTON_WIDTH and
                button_y <= mouse_y <= button_y + BUTTON_HEIGHT)
    
    def is_skip_button_clicked(self, pos):
        """Check if the skip turn button was clicked"""
        mouse_x, mouse_y = pos
        
        button_x = SCREEN_WIDTH / 2 + 20
        button_y = SCREEN_HEIGHT - 60
        
        return (button_x <= mouse_x <= button_x + BUTTON_WIDTH and
                button_y <= mouse_y <= button_y + BUTTON_HEIGHT)
    
    def can_bear_off(self):
        """Check if the current player can bear off"""
        # Check if any valid move is a bearing off move
        return self.selected_point is not None and 'off' in self.valid_destination_points
    
    def draw(self):
        """Draw the game board and UI"""
        # Fill background with wood texture color
        self.screen.fill((233, 196, 150))  # Light wood color
        
        # Draw board
        self.draw_board()
        
        # Draw checkers
        self.draw_checkers()
        
        # Draw dragged checker if dragging
        if self.dragging and self.drag_source is not None:
            self.draw_dragged_checker()
        
        # Draw dice
        self.draw_dice()
        
        # Draw dice input
        self.draw_dice_input()
        
        # Draw buttons
        self.draw_buttons()
        
        # Draw message
        self.draw_message()
        
        pygame.display.flip()
    
    def draw_board(self):
        """Draw the Narde board"""
        # Calculate board position
        board_left = (SCREEN_WIDTH - (12 * POINT_WIDTH + CENTER_BAR_WIDTH)) / 2
        board_top = (SCREEN_HEIGHT - POINT_HEIGHT * 2) / 2
        
        # Draw board background
        pygame.draw.rect(self.screen, BROWN, 
                         (board_left, board_top, 
                          12 * POINT_WIDTH + CENTER_BAR_WIDTH + POINT_WIDTH, # Add space for bearing off
                          POINT_HEIGHT * 2))
        
        # Draw center bar
        pygame.draw.rect(self.screen, DARK_BROWN, 
                         (board_left + 6 * POINT_WIDTH, board_top, 
                          CENTER_BAR_WIDTH, POINT_HEIGHT * 2))
        
        # Get board in current player's perspective
        board = self.env.game.get_perspective_board(self.current_player)
        
        # Draw point triangles
        for point_idx, (x, y, is_top_row) in self.point_coordinates.items():
            # Alternate colors for the points
            even_point = point_idx % 2 == 0
            # Determine point color based on state and position
            if self.selected_point == point_idx:
                point_color = GOLD
            elif point_idx in self.valid_destination_points:
                point_color = LIGHT_GREEN
            else:
                point_color = LIGHT_BROWN if even_point else BROWN
            
            # Draw the triangular point
            if is_top_row:
                # Draw top row triangles (pointing down)
                points = [
                    (x + POINT_WIDTH/2, y + POINT_TRIANGLE_HEIGHT),  # Tip
                    (x, y),                                          # Left corner
                    (x + POINT_WIDTH, y)                             # Right corner
                ]
            else:
                # Draw bottom row triangles (pointing up)
                points = [
                    (x + POINT_WIDTH/2, y + POINT_HEIGHT - POINT_TRIANGLE_HEIGHT),  # Tip
                    (x, y + POINT_HEIGHT),                                         # Left corner
                    (x + POINT_WIDTH, y + POINT_HEIGHT)                            # Right corner
                ]
            
            pygame.draw.polygon(self.screen, point_color, points)
            
            # Draw point outline
            pygame.draw.polygon(self.screen, BLACK, points, 1)
            
            # Draw point index (adding 1 to convert from 0-based to 1-based position numbers)
            game_position = point_idx + 1
            text = self.small_font.render(str(game_position), True, BLACK)
            
            # Special highlighting for key positions - from current player's perspective
            if (self.current_player == 1 and game_position == 24) or \
               (self.current_player == -1 and game_position == 12):  # Current player's head
                pygame.draw.circle(self.screen, GREEN, 
                                  (x + POINT_WIDTH/2, y + POINT_HEIGHT - 15 if not is_top_row else y + 15), 12)
            elif (self.current_player == 1 and game_position == 12) or \
                 (self.current_player == -1 and game_position == 24):  # Opponent's head
                pygame.draw.circle(self.screen, RED, 
                                  (x + POINT_WIDTH/2, y + POINT_HEIGHT - 15 if not is_top_row else y + 15), 12)
                
            # Position the text
            if is_top_row:
                text_rect = text.get_rect(center=(x + POINT_WIDTH/2, y + 15))
            else:
                text_rect = text.get_rect(center=(x + POINT_WIDTH/2, y + POINT_HEIGHT - 15))
            self.screen.blit(text, text_rect)
        
        # Draw bearing off area - always on the right side from current player's perspective
        bearing_off_x = board_left + CENTER_BAR_WIDTH + 12 * POINT_WIDTH
        bearing_off_y = board_top
        pygame.draw.rect(self.screen, DARK_BROWN, 
                         (bearing_off_x, bearing_off_y, POINT_WIDTH, POINT_HEIGHT * 2))
        
        # Draw "BEAR OFF" text
        bear_text = self.small_font.render("BEAR", True, WHITE)
        off_text = self.small_font.render("OFF", True, WHITE)
        
        bear_rect = bear_text.get_rect(center=(bearing_off_x + POINT_WIDTH/2, bearing_off_y + POINT_HEIGHT/2 - 30))
        off_rect = off_text.get_rect(center=(bearing_off_x + POINT_WIDTH/2, bearing_off_y + POINT_HEIGHT/2 + 30))
        
        self.screen.blit(bear_text, bear_rect)
        self.screen.blit(off_text, off_rect)
        
        # Draw a helpful "HOME" indicator for positions 1-6 from current player's perspective
        home_text = self.small_font.render("HOME", True, GREEN)
        home_rect = home_text.get_rect(center=(board_left + CENTER_BAR_WIDTH + 9 * POINT_WIDTH, 
                                              board_top + POINT_HEIGHT - 15))
        pygame.draw.rect(self.screen, (255, 255, 255, 180), 
                        (home_rect.left - 5, home_rect.top - 5, 
                         home_rect.width + 10, home_rect.height + 10))
        self.screen.blit(home_text, home_rect)
    
    def draw_checkers(self):
        """Draw checkers on the board"""
        # Get the actual board state
        board = self.env.game.board
        
        # Skip drawing the one being dragged
        skip_point = self.drag_source if self.dragging else None
        
        # Draw checkers for each point
        for point_idx, (x, y, is_top_row) in self.point_coordinates.items():
            # Skip if this is the point being dragged from
            if point_idx == skip_point:
                # Reduce checker count by 1 for display
                checkers = board[point_idx]
                if abs(checkers) <= 1:
                    continue  # Skip entirely if only one checker
                # Otherwise, draw with one less
                checkers = checkers - 1 if checkers > 0 else checkers + 1
            else:
                checkers = board[point_idx]
            
            # Skip if no checkers
            if checkers == 0:
                continue
            
            # Determine checker color based on the actual board state
            # White checkers are positive numbers, Black checkers are negative numbers
            checker_color = WHITE if checkers > 0 else BLACK
            num_checkers = abs(checkers)
            
            # Calculate stack direction and positions
            if is_top_row:
                # Stack down from top
                start_y = y + 25
                direction = 1
            else:
                # Stack up from bottom
                start_y = y + POINT_HEIGHT - 25
                direction = -1
            
            # Draw stack of checkers
            stack_height = min(7, num_checkers)  # Limit stacking display
            for j in range(stack_height):
                center_x = x + POINT_WIDTH / 2
                center_y = start_y + (j * direction * 15)
                
                # Draw checker with shadow effect
                pygame.draw.circle(self.screen, (50, 50, 50), (center_x+2, center_y+2), CHECKER_SIZE/2)  # Shadow
                pygame.draw.circle(self.screen, checker_color, (center_x, center_y), CHECKER_SIZE/2)
                pygame.draw.circle(self.screen, (100, 100, 100) if checker_color == BLACK else (200, 200, 200), 
                                  (center_x, center_y), CHECKER_SIZE/2 - 3)  # Inner ring
                pygame.draw.circle(self.screen, BLACK, (center_x, center_y), CHECKER_SIZE/2, 1)  # Outline
            
            # Show count if more than the display limit
            if num_checkers > stack_height:
                text_color = BLACK if checker_color == WHITE else WHITE
                text = self.small_font.render(str(num_checkers), True, text_color)
                if is_top_row:
                    text_pos = (center_x, start_y + (stack_height-1) * direction * 15 + 20)
                else:
                    text_pos = (center_x, start_y + (stack_height-1) * direction * 15 - 20)
                text_rect = text.get_rect(center=text_pos)
                self.screen.blit(text, text_rect)
    
    def draw_dragged_checker(self):
        """Draw the checker being dragged"""
        if not self.dragging or self.drag_source is None:
            return
        
        # Get board from environment based on current player's perspective
        board = self.env.game.get_perspective_board(self.current_player)
        
        # Determine checker color based on the source point
        checkers = board[self.drag_source]
        checker_color = WHITE if checkers > 0 else BLACK
        
        # Draw at mouse position
        center_x, center_y = self.drag_pos
        
        # Draw with shadow and highlight effect
        pygame.draw.circle(self.screen, (50, 50, 50), (center_x+3, center_y+3), CHECKER_SIZE/2+2)  # Shadow
        pygame.draw.circle(self.screen, checker_color, (center_x, center_y), CHECKER_SIZE/2+2)  # Slightly larger
        pygame.draw.circle(self.screen, (100, 100, 100) if checker_color == BLACK else (200, 200, 200), 
                          (center_x, center_y), CHECKER_SIZE/2 - 2)  # Inner ring
        # Add highlight effect for dragged checker
        pygame.draw.circle(self.screen, GOLD, (center_x, center_y), CHECKER_SIZE/2+2, 2)  # Golden outline
    
    def draw_dice(self):
        """Draw dice"""
        # Center the dice in the middle bar
        board_left = (SCREEN_WIDTH - (12 * POINT_WIDTH + CENTER_BAR_WIDTH)) / 2
        center_x = board_left + 6 * POINT_WIDTH + CENTER_BAR_WIDTH / 2
        dice_y = SCREEN_HEIGHT / 2 + 50
        
        # Draw dice panel background
        panel_width = 180
        panel_height = 100
        pygame.draw.rect(self.screen, (255, 255, 255, 180), 
                        (center_x - panel_width/2, dice_y - panel_height/2, 
                         panel_width, panel_height))
        
        for i, (die, used) in enumerate(zip(self.dice, self.dice_used)):
            # Position dice horizontally centered, with some spacing
            dice_offset = -50 if i == 0 else 50
            die_x = center_x + dice_offset - DICE_SIZE/2
            
            # Draw die background with shadow
            color = GRAY if used else WHITE
            
            # Shadow
            pygame.draw.rect(self.screen, (50, 50, 50), 
                            (die_x + 3, dice_y - DICE_SIZE/2 + 3, DICE_SIZE, DICE_SIZE), 
                            border_radius=8)
            
            # Die body
            pygame.draw.rect(self.screen, color, 
                            (die_x, dice_y - DICE_SIZE/2, DICE_SIZE, DICE_SIZE), 
                            border_radius=8)
            
            # Die border
            pygame.draw.rect(self.screen, BLACK, 
                            (die_x, dice_y - DICE_SIZE/2, DICE_SIZE, DICE_SIZE), 
                            2, border_radius=8)
            
            # If dice have been rolled, draw the value
            if self.roll_entered:
                # Draw dice pips based on value
                self.draw_die_pips(die_x, dice_y - DICE_SIZE/2, die)
            else:
                # If not rolled, show a question mark
                text = self.large_font.render("?", True, BLACK)
                text_rect = text.get_rect(center=(die_x + DICE_SIZE/2, dice_y))
                self.screen.blit(text, text_rect)
    
    def draw_die_pips(self, x, y, value):
        """Draw pips on a die based on its value"""
        # Pip positions (relative to die top-left corner)
        pip_positions = {
            1: [(DICE_SIZE/2, DICE_SIZE/2)],  # Center
            2: [(DICE_SIZE/4, DICE_SIZE/4), (3*DICE_SIZE/4, 3*DICE_SIZE/4)],  # Top-left, bottom-right
            3: [(DICE_SIZE/4, DICE_SIZE/4), (DICE_SIZE/2, DICE_SIZE/2), (3*DICE_SIZE/4, 3*DICE_SIZE/4)],  # TL, center, BR
            4: [(DICE_SIZE/4, DICE_SIZE/4), (3*DICE_SIZE/4, DICE_SIZE/4), 
               (DICE_SIZE/4, 3*DICE_SIZE/4), (3*DICE_SIZE/4, 3*DICE_SIZE/4)],  # All corners
            5: [(DICE_SIZE/4, DICE_SIZE/4), (3*DICE_SIZE/4, DICE_SIZE/4), 
               (DICE_SIZE/2, DICE_SIZE/2),  # Four corners + center
               (DICE_SIZE/4, 3*DICE_SIZE/4), (3*DICE_SIZE/4, 3*DICE_SIZE/4)],
            6: [(DICE_SIZE/4, DICE_SIZE/4), (3*DICE_SIZE/4, DICE_SIZE/4), 
               (DICE_SIZE/4, DICE_SIZE/2), (3*DICE_SIZE/4, DICE_SIZE/2),  # 2 rows of 3
               (DICE_SIZE/4, 3*DICE_SIZE/4), (3*DICE_SIZE/4, 3*DICE_SIZE/4)]
        }
        
        # Draw pips for the current value
        positions = pip_positions.get(value, [])
        for pip_x, pip_y in positions:
            pygame.draw.circle(self.screen, BLACK, (x + pip_x, y + pip_y), 6)
    
    def draw_dice_input(self):
        """Draw dice input fields"""
        if self.roll_entered:
            return
        
        # Draw text instructing to enter dice values
        instruction_text = self.font.render("Enter dice values (1-6):", True, BLACK)
        instruction_rect = instruction_text.get_rect(topleft=(SCREEN_WIDTH/2 - 180, SCREEN_HEIGHT - 150))
        self.screen.blit(instruction_text, instruction_rect)
        
        # Draw input boxes
        dice_x = SCREEN_WIDTH / 2 - 100
        dice_y = SCREEN_HEIGHT - 120
        
        for i in range(2):
            # Draw input box with shadow effect
            # Shadow
            pygame.draw.rect(self.screen, (100, 100, 100), 
                            (dice_x + i * 120 + 2, dice_y + 2, 80, 40))
            
            # Box
            color = RED if self.active_dice_input == i else BLACK
            pygame.draw.rect(self.screen, WHITE, 
                            (dice_x + i * 120, dice_y, 80, 40))
            pygame.draw.rect(self.screen, color, 
                            (dice_x + i * 120, dice_y, 80, 40), 2)
            
            # Draw input value
            text = self.large_font.render(self.dice_input[i] or "_", True, BLACK)
            text_rect = text.get_rect(center=(dice_x + i * 120 + 40, dice_y + 20))
            self.screen.blit(text, text_rect)
            
            # Label the dice
            label = self.small_font.render(f"Die {i+1}", True, BLACK)
            label_rect = label.get_rect(center=(dice_x + i * 120 + 40, dice_y - 15))
            self.screen.blit(label, label_rect)
    
    def draw_buttons(self):
        """Draw UI buttons"""
        button_y = SCREEN_HEIGHT - 60
        
        # Roll button
        roll_x = SCREEN_WIDTH / 2 - BUTTON_WIDTH - 20
        
        # Button shadow
        pygame.draw.rect(self.screen, (50, 50, 50), 
                        (roll_x + 3, button_y + 3, BUTTON_WIDTH, BUTTON_HEIGHT),
                        border_radius=5)
        
        # Button body
        color = GREEN if not self.roll_entered else GRAY
        pygame.draw.rect(self.screen, color, 
                        (roll_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT),
                        border_radius=5)
        
        # Button outline
        pygame.draw.rect(self.screen, BLACK, 
                        (roll_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT), 
                        2, border_radius=5)
        
        text = self.large_font.render("Roll Dice", True, BLACK if not self.roll_entered else (50, 50, 50))
        text_rect = text.get_rect(center=(roll_x + BUTTON_WIDTH/2, button_y + BUTTON_HEIGHT/2))
        self.screen.blit(text, text_rect)
        
        # Skip turn button
        skip_x = SCREEN_WIDTH / 2 + 20
        
        # Button shadow
        pygame.draw.rect(self.screen, (50, 50, 50), 
                        (skip_x + 3, button_y + 3, BUTTON_WIDTH, BUTTON_HEIGHT),
                        border_radius=5)
        
        # Button body
        color = BLUE if self.roll_entered else GRAY
        pygame.draw.rect(self.screen, color, 
                        (skip_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT),
                        border_radius=5)
        
        # Button outline
        pygame.draw.rect(self.screen, BLACK, 
                        (skip_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT), 
                        2, border_radius=5)
        
        text = self.large_font.render("Skip Turn", True, BLACK if self.roll_entered else (50, 50, 50))
        text_rect = text.get_rect(center=(skip_x + BUTTON_WIDTH/2, button_y + BUTTON_HEIGHT/2))
        self.screen.blit(text, text_rect)
    
    def draw_message(self):
        """Draw status message"""
        # Draw message box
        message_box_width = 500
        message_box_height = 60
        message_x = SCREEN_WIDTH/2 - message_box_width/2
        message_y = 30
        
        # Box background with shadow
        pygame.draw.rect(self.screen, (50, 50, 50), 
                        (message_x + 3, message_y + 3, message_box_width, message_box_height),
                        border_radius=10)
        
        pygame.draw.rect(self.screen, (255, 255, 255, 200), 
                        (message_x, message_y, message_box_width, message_box_height),
                        border_radius=10)
        
        # Draw status message
        text = self.font.render(self.message, True, BLACK)
        text_rect = text.get_rect(center=(SCREEN_WIDTH/2, message_y + 20))
        self.screen.blit(text, text_rect)
        
        # Draw current player indicator
        player_text = f"Current Player: {self.current_player_name()}"
        text = self.font.render(player_text, True, BLACK)
        text_rect = text.get_rect(center=(SCREEN_WIDTH/2, message_y + 45))
        self.screen.blit(text, text_rect)
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        
        while True:
            for event in pygame.event.get():
                self.handle_event(event)
            
            self.draw()
            clock.tick(30)

if __name__ == "__main__":
    game = LongNardeGame()
    game.run()