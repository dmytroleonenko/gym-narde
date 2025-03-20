#!/usr/bin/env python3
"""
Benchmarking framework for Narde agents.

This framework allows running game simulations between any two Narde agents
and collects statistics on the outcomes. It supports running games with
fixed colors or with alternating colors.
"""

import gym_narde  # Import this first to register the environment
import gymnasium as gym
import numpy as np
import random
import time
import importlib
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod


class ColorAssignment(Enum):
    """Enum to specify how agents should be assigned colors."""
    ALTERNATE = auto()  # Agents alternate between White and Black
    FIXED = auto()      # Agents keep the same color for all games


@dataclass
class GameResult:
    """Dataclass to store the result of a single game."""
    white_agent_name: str
    black_agent_name: str
    winner: Optional[str]  # 'white', 'black', or None for draw
    steps: int
    white_borne_off: int
    black_borne_off: int
    truncated: bool


@dataclass
class BenchmarkResult:
    """Dataclass to store the results of a benchmark run."""
    agent1_name: str
    agent2_name: str
    total_games: int
    agent1_wins: int
    agent2_wins: int
    draws: int
    agent1_as_white_wins: int
    agent1_as_white_games: int
    agent1_as_black_wins: int
    agent1_as_black_games: int
    white_wins: int  # Total wins by White (regardless of agent)
    black_wins: int  # Total wins by Black (regardless of agent)
    avg_steps: float
    max_steps: int
    avg_white_borne_off: float
    avg_black_borne_off: float
    game_results: List[GameResult]
    
    @property
    def agent1_win_rate(self) -> float:
        """Return the win rate for agent1."""
        return self.agent1_wins / self.total_games if self.total_games > 0 else 0
    
    @property
    def agent2_win_rate(self) -> float:
        """Return the win rate for agent2."""
        return self.agent2_wins / self.total_games if self.total_games > 0 else 0
    
    @property
    def draw_rate(self) -> float:
        """Return the draw rate."""
        return self.draws / self.total_games if self.total_games > 0 else 0
    
    @property
    def agent1_as_white_win_rate(self) -> float:
        """Return the win rate for agent1 when playing as White."""
        return (
            self.agent1_as_white_wins / self.agent1_as_white_games 
            if self.agent1_as_white_games > 0 else 0
        )
    
    @property
    def agent1_as_black_win_rate(self) -> float:
        """Return the win rate for agent1 when playing as Black."""
        return (
            self.agent1_as_black_wins / self.agent1_as_black_games 
            if self.agent1_as_black_games > 0 else 0
        )
    
    @property
    def white_win_rate(self) -> float:
        """Return the win rate for White color."""
        return self.white_wins / self.total_games if self.total_games > 0 else 0
    
    @property
    def black_win_rate(self) -> float:
        """Return the win rate for Black color."""
        return self.black_wins / self.total_games if self.total_games > 0 else 0


class GenericNardeAgent(ABC):
    """
    Abstract base class that all Narde agents should inherit from.
    Defines the interface that must be implemented by any agent.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the agent."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the agent's state for a new game."""
        pass
    
    @abstractmethod
    def select_action(self, env, state):
        """
        Select an action based on the current state.
        
        Args:
            env: The environment
            state: The current state observation
            
        Returns:
            The selected action
        """
        pass


class NardeBenchmark:
    """
    Framework for benchmarking Narde agents against each other.
    
    This class runs simulations between two agents and collects statistics
    on their performance. It can run games with agents in fixed colors
    or with alternating colors.
    """
    
    def __init__(self, max_steps_per_game: int = 500, verbose: bool = False, debug_episodes: int = 0):
        """
        Initialize the benchmark framework.
        
        Args:
            max_steps_per_game: Maximum number of steps per game before truncation
            verbose: Whether to print progress information
            debug_episodes: Number of episodes to print detailed debug info for
        """
        self.max_steps_per_game = max_steps_per_game
        self.verbose = verbose
        self.debug_episodes = debug_episodes
        
        # Create the environment
        self.env = gym.make('Narde-v0')
        self.env_unwrapped = self.env.unwrapped
    
    def run_single_game(self, white_agent, black_agent, episode: int = 0) -> GameResult:
        """
        Run a single game between two agents with fixed colors.
        
        Args:
            white_agent: Agent that will play as White
            black_agent: Agent that will play as Black
            episode: Episode number for debugging
            
        Returns:
            GameResult: The result of the game
        """
        # Reset the environment and get initial state
        state, info = self.env.reset()
        
        # Reset agents for a new episode
        white_agent.reset()
        black_agent.reset()
        
        step_count = 0
        current_player_is_white = True  # Game always starts with White
        debug_this_episode = episode < self.debug_episodes
        
        # Track which agent is currently active (starts with white_agent)
        current_agent = white_agent
        
        if debug_this_episode:
            print(f"\nEpisode {episode+1} starting")
            print(f"White agent: {white_agent.name}")
            print(f"Black agent: {black_agent.name}")
            print(f"Initial board state: {self.env_unwrapped.game.board}")
            print(f"Initial dice: {self.env_unwrapped.dice}")
        
        for step in range(self.max_steps_per_game):
            # Get valid moves
            valid_moves = self.env_unwrapped.game.get_valid_moves()
            
            if debug_this_episode:
                print(f"Step {step}, Current player: {'White' if current_player_is_white else 'Black'}")
                print(f"Current agent: {current_agent.name}")
                print(f"Board: {self.env_unwrapped.game.board}")
                print(f"Dice: {self.env_unwrapped.dice}")
                print(f"Valid moves: {valid_moves}")
                print(f"White borne off: {self.env_unwrapped.game.borne_off_white}")
                print(f"Black borne off: {self.env_unwrapped.game.borne_off_black}")
            
            if not valid_moves:
                # If no valid moves, clear dice and re-roll for next player
                self.env_unwrapped.dice = []  # Clear existing dice
                
                if debug_this_episode:
                    print(f"No valid moves, skipping turn")
                
                # Re-roll the dice for the next player
                if hasattr(self.env_unwrapped, "_roll_dice"):
                    self.env_unwrapped._roll_dice()
                else:
                    # Fallback to manually rolling dice
                    d1 = random.randint(1, 6)
                    d2 = random.randint(1, 6)
                    if d1 == d2:
                        self.env_unwrapped.dice = [d1, d1, d1, d1]  # Double dice give four moves
                    else:
                        self.env_unwrapped.dice = [d1, d2]
                
                # Rotate board for next player
                self.env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_white = not current_player_is_white
                # Switch the current agent
                current_agent = black_agent if current_agent == white_agent else white_agent
                
                if debug_this_episode:
                    print(f"Board after rotation: {self.env_unwrapped.game.board}")
                    print(f"New dice: {self.env_unwrapped.dice}")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'}")
                    print(f"Next agent: {current_agent.name}")
                
                continue
            
            # Get action from the current agent
            try:
                action = current_agent.select_action(
                    self.env_unwrapped, state
                )
            except ValueError as e:
                # If agent can't select an action, log the error and end the game
                print(f"Error: {e}")
                print(f"Agent {current_agent.name} failed to select action")
                return GameResult(
                    white_agent_name=white_agent.name,
                    black_agent_name=black_agent.name,
                    winner=None,  # Draw due to error
                    steps=step_count,
                    white_borne_off=self.env_unwrapped.game.borne_off_white,
                    black_borne_off=self.env_unwrapped.game.borne_off_black,
                    truncated=True
                )
            
            if debug_this_episode:
                # Convert action back to move for debugging
                move_index, move_type = action
                if move_type == 1:
                    # Bear off move
                    from_pos = move_index // 24
                    move = (from_pos, 'off')
                else:
                    # Regular move
                    from_pos = move_index // 24
                    to_pos = move_index % 24
                    move = (from_pos, to_pos)
                print(f"Agent {current_agent.name} selected move: {move}")
            
            # Take the action
            next_state, reward, done, truncated, info = self.env.step(action)
            
            if debug_this_episode:
                print(f"Reward: {reward}")
                print(f"Done: {done}")
                print(f"Board after move: {self.env_unwrapped.game.board}")
                print(f"White borne off after move: {self.env_unwrapped.game.borne_off_white}")
                print(f"Black borne off after move: {self.env_unwrapped.game.borne_off_black}")
                print("-" * 50)
            
            state = next_state
            step_count += 1
            
            # Track dice state before checking for rotation
            dice_before = self.env_unwrapped.dice.copy() if self.env_unwrapped.dice else []
            
            # Check if the environment automatically rotated the board inside step()
            auto_rotated = False
            if "skipped_turn" in info or "skipped_multiple_turns" in info:
                auto_rotated = True
                current_player_is_white = not current_player_is_white
                # Switch the current agent
                current_agent = black_agent if current_agent == white_agent else white_agent
                
                if debug_this_episode:
                    print("Environment automatically rotated board due to skipped turn")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'}")
                    print(f"Next agent: {current_agent.name}")
            
            # Check if dice were depleted and new dice rolled
            elif (len(dice_before) > 0 and len(dice_before) <= 2 and 
                  len(self.env_unwrapped.dice) >= 2 and len(self.env_unwrapped.dice) > len(dice_before)):
                auto_rotated = True
                current_player_is_white = not current_player_is_white
                # Switch the current agent
                current_agent = black_agent if current_agent == white_agent else white_agent
                
                if debug_this_episode:
                    print("Environment automatically rotated board due to dice depletion")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'}")
                    print(f"Next agent: {current_agent.name}")
            
            if done:
                # In the environment, the perspective is always White, so if game is done,
                # the current_agent won, regardless of which color they were playing
                winner_agent = current_agent
                winner_color = 'white' if winner_agent == white_agent else 'black'
                
                if debug_this_episode:
                    print(f"Game over: {winner_agent.name} won!")
                    print(f"Winner was playing as: {winner_color}")
                    print(f"Final board: {self.env_unwrapped.game.board}")
                    print(f"White borne off: {self.env_unwrapped.game.borne_off_white}")
                    print(f"Black borne off: {self.env_unwrapped.game.borne_off_black}")
                
                return GameResult(
                    white_agent_name=white_agent.name,
                    black_agent_name=black_agent.name,
                    winner=winner_color,
                    steps=step_count,
                    white_borne_off=self.env_unwrapped.game.borne_off_white,
                    black_borne_off=self.env_unwrapped.game.borne_off_black,
                    truncated=False
                )
            
            if truncated:
                if debug_this_episode:
                    print(f"Episode truncated after {step_count} steps")
                
                return GameResult(
                    white_agent_name=white_agent.name,
                    black_agent_name=black_agent.name,
                    winner=None,  # Draw due to truncation
                    steps=step_count,
                    white_borne_off=self.env_unwrapped.game.borne_off_white,
                    black_borne_off=self.env_unwrapped.game.borne_off_black,
                    truncated=True
                )
            
            # Rotate board for next player if needed - only if env didn't already do it
            if not (done or truncated) and not self.env_unwrapped.dice and not auto_rotated:
                self.env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_white = not current_player_is_white
                # Switch the current agent
                current_agent = black_agent if current_agent == white_agent else white_agent
                
                if debug_this_episode:
                    print(f"Board rotated for next player")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'}")
                    print(f"Next agent: {current_agent.name}")
        
        # If we reach here, the game was truncated due to reaching max_steps
        return GameResult(
            white_agent_name=white_agent.name,
            black_agent_name=black_agent.name,
            winner=None,  # Draw due to max steps
            steps=step_count,
            white_borne_off=self.env_unwrapped.game.borne_off_white,
            black_borne_off=self.env_unwrapped.game.borne_off_black,
            truncated=True
        )
    
    def run_benchmark(self, agent1, agent2, num_games=100, 
                     color_assignment=ColorAssignment.ALTERNATE) -> BenchmarkResult:
        """
        Run a benchmark of games between two agents.
        
        Args:
            agent1: First agent
            agent2: Second agent
            num_games: Number of games to run
            color_assignment: How to assign colors to agents
                
        Returns:
            BenchmarkResult: Statistics from the benchmark run
        """
        start_time = time.time()
        
        # Initialize tracking metrics
        agent1_wins = 0
        agent2_wins = 0
        draws = 0
        
        agent1_as_white_games = 0
        agent1_as_white_wins = 0
        agent1_as_black_games = 0
        agent1_as_black_wins = 0
        
        # Track wins by color (regardless of agent)
        white_wins = 0
        black_wins = 0
        
        steps_list = []
        white_borne_off_list = []
        black_borne_off_list = []
        
        game_results = []
        
        if self.verbose:
            print(f"Starting benchmark: {agent1.name} vs {agent2.name}")
            print(f"Number of games: {num_games}")
            print(f"Color assignment: {color_assignment.name}")
            print("-" * 60)
        
        for game_idx in range(num_games):
            # Determine which agent plays which color
            if color_assignment == ColorAssignment.ALTERNATE:
                # Alternate colors every game
                if game_idx % 2 == 0:
                    white_agent, black_agent = agent1, agent2
                    white_is_agent1 = True
                else:
                    white_agent, black_agent = agent2, agent1
                    white_is_agent1 = False
            else:
                # Fixed colors for all games
                white_agent, black_agent = agent1, agent2
                white_is_agent1 = True
            
            # Update color tracking
            if white_is_agent1:
                agent1_as_white_games += 1
            else:
                agent1_as_black_games += 1
                
            # Run a single game
            result = self.run_single_game(white_agent, black_agent, game_idx)
            game_results.append(result)
            
            # Update metrics
            steps_list.append(result.steps)
            white_borne_off_list.append(result.white_borne_off)
            black_borne_off_list.append(result.black_borne_off)
            
            # Track wins based on the winner
            if result.winner is None:
                draws += 1
            elif result.winner == 'white':
                white_wins += 1  # Increment white wins
                if white_is_agent1:
                    agent1_wins += 1
                    agent1_as_white_wins += 1
                else:
                    agent2_wins += 1
            elif result.winner == 'black':
                black_wins += 1  # Increment black wins
                if not white_is_agent1:
                    agent1_wins += 1
                    agent1_as_black_wins += 1
                else:
                    agent2_wins += 1
            
            # Print progress
            if self.verbose and (game_idx + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                games_completed = game_idx + 1
                games_per_second = games_completed / elapsed_time if elapsed_time > 0 else 0
                estimated_remaining = (num_games - games_completed) / games_per_second if games_per_second > 0 else 0
                
                print(f"Completed {games_completed}/{num_games} games ({games_per_second:.2f} games/sec)")
                print(f"  {agent1.name} win rate: {agent1_wins/games_completed:.2%}")
                print(f"  {agent2.name} win rate: {agent2_wins/games_completed:.2%}")
                print(f"  Draws: {draws/games_completed:.2%}")
                print(f"  Average steps: {sum(steps_list)/games_completed:.2f}")
                print(f"  Estimated time remaining: {estimated_remaining:.1f} seconds")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if self.verbose:
            print("\nBenchmark complete")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average time per game: {total_time/num_games:.4f} seconds")
        
        # Create and return the final result
        return BenchmarkResult(
            agent1_name=agent1.name,
            agent2_name=agent2.name,
            total_games=num_games,
            agent1_wins=agent1_wins,
            agent2_wins=agent2_wins,
            draws=draws,
            agent1_as_white_wins=agent1_as_white_wins,
            agent1_as_white_games=agent1_as_white_games,
            agent1_as_black_wins=agent1_as_black_wins,
            agent1_as_black_games=agent1_as_black_games,
            white_wins=white_wins,
            black_wins=black_wins,
            avg_steps=sum(steps_list)/num_games if num_games > 0 else 0,
            max_steps=max(steps_list) if steps_list else 0,
            avg_white_borne_off=sum(white_borne_off_list)/num_games if num_games > 0 else 0,
            avg_black_borne_off=sum(black_borne_off_list)/num_games if num_games > 0 else 0,
            game_results=game_results
        )
    
    def print_benchmark_results(self, results: BenchmarkResult):
        """
        Print detailed results from a benchmark run.
        
        Args:
            results: The BenchmarkResult to print
        """
        print("\n" + "=" * 60)
        print(f"Benchmark Results: {results.agent1_name} vs {results.agent2_name}")
        print("=" * 60)
        print(f"Total Games: {results.total_games}")
        print(f"{results.agent1_name} Win Rate: {results.agent1_win_rate:.2%} ({results.agent1_wins} wins)")
        print(f"{results.agent2_name} Win Rate: {results.agent2_win_rate:.2%} ({results.agent2_wins} wins)")
        print(f"Draws: {results.draw_rate:.2%} ({results.draws} draws)")
        print("")
        
        # Print color-based statistics
        print(f"White Color Win Rate: {results.white_win_rate:.2%} ({results.white_wins}/{results.total_games})")
        print(f"Black Color Win Rate: {results.black_win_rate:.2%} ({results.black_wins}/{results.total_games})")
        
        # Print agent-specific color stats for both agents
        if results.agent1_as_white_games > 0:
            print(f"{results.agent1_name} as White: {results.agent1_as_white_win_rate:.2%} " +
                 f"({results.agent1_as_white_wins}/{results.agent1_as_white_games})")
        
        if results.agent1_as_black_games > 0:
            print(f"{results.agent1_name} as Black: {results.agent1_as_black_win_rate:.2%} " +
                 f"({results.agent1_as_black_wins}/{results.agent1_as_black_games})")
        
        # Add agent2 color-specific stats
        agent2_as_white_games = results.total_games - results.agent1_as_white_games
        agent2_as_white_wins = results.white_wins - results.agent1_as_white_wins
        agent2_as_black_games = results.total_games - results.agent1_as_black_games
        agent2_as_black_wins = results.black_wins - results.agent1_as_black_wins
        
        if agent2_as_white_games > 0:
            agent2_as_white_win_rate = agent2_as_white_wins / agent2_as_white_games
            print(f"{results.agent2_name} as White: {agent2_as_white_win_rate:.2%} " +
                 f"({agent2_as_white_wins}/{agent2_as_white_games})")
        
        if agent2_as_black_games > 0:
            agent2_as_black_win_rate = agent2_as_black_wins / agent2_as_black_games
            print(f"{results.agent2_name} as Black: {agent2_as_black_win_rate:.2%} " +
                 f"({agent2_as_black_wins}/{agent2_as_black_games})")
        
        print("")
        print(f"Average Steps Per Game: {results.avg_steps:.2f}")
        print(f"Maximum Steps in a Game: {results.max_steps}")
        print(f"Average White Checkers Borne Off: {results.avg_white_borne_off:.2f}")
        print(f"Average Black Checkers Borne Off: {results.avg_black_borne_off:.2f}")
        
        # Calculate average steps to bear off each checker
        total_borne_off = results.avg_white_borne_off + results.avg_black_borne_off
        if total_borne_off > 0:
            avg_steps_per_checker = results.avg_steps / total_borne_off
            print(f"Average Steps Per Checker Borne Off: {avg_steps_per_checker:.2f}")
        
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    import argparse
    import sys
    import importlib
    
    parser = argparse.ArgumentParser(description='Run benchmarks for Narde agents')
    parser.add_argument('--agent1', type=str, default='narde_agents.RandomAgent', 
                        help='First agent class (module.ClassName format)')
    parser.add_argument('--agent2', type=str, default='narde_agents.RandomAgent',
                        help='Second agent class (module.ClassName format)')
    parser.add_argument('--model', type=str, default='narde_dqn_model.pth',
                        help='Path to DQN model file (if using DQN agent)')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games to run')
    parser.add_argument('--color_assignment', type=str, default='alternate',
                        choices=['alternate', 'fixed'], 
                        help='Color assignment strategy')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per game')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    parser.add_argument('--debug', type=int, default=0,
                        help='Number of episodes to print debug info for')
    parser.add_argument('--agent1_args', type=str, default='',
                        help='Additional arguments for agent1 (comma-separated key=value pairs)')
    parser.add_argument('--agent2_args', type=str, default='',
                        help='Additional arguments for agent2 (comma-separated key=value pairs)')
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    def create_agent(agent_class_path, agent_args_str, model_path=None, name_suffix=""):
        """
        Dynamically create an agent from its class path.
        
        Args:
            agent_class_path: String in format 'module.ClassName'
            agent_args_str: String of comma-separated key=value pairs
            model_path: Optional path to a model file
            name_suffix: Suffix to append to the agent's name
        
        Returns:
            An instance of the specified agent class
        """
        try:
            # Parse module and class names
            module_path, class_name = agent_class_path.rsplit('.', 1)
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the class
            agent_class = getattr(module, class_name)
            
            # Parse additional arguments
            agent_args = {}
            if agent_args_str:
                for arg_pair in agent_args_str.split(','):
                    if '=' in arg_pair:
                        key, value = arg_pair.split('=', 1)
                        # Try to convert to appropriate types
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.isdigit():
                            value = int(value)
                        elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
                            value = float(value)
                        agent_args[key.strip()] = value
            
            # Handle special cases for known agent types
            if class_name == 'DQNAgent' and model_path:
                agent_args['model_path'] = model_path
            
            # Set name if not provided
            if 'name' not in agent_args:
                agent_args['name'] = f"{class_name}{name_suffix}"
            
            # Create and return the agent
            return agent_class(**agent_args)
        
        except (ImportError, AttributeError, ValueError) as e:
            print(f"Error creating agent {agent_class_path}: {e}")
            sys.exit(1)
    
    # Create agents based on command line arguments
    agent1 = create_agent(args.agent1, args.agent1_args, args.model, "1")
    agent2 = create_agent(args.agent2, args.agent2_args, args.model, "2")
    
    # Determine color assignment
    color_assignment = (ColorAssignment.ALTERNATE if args.color_assignment == 'alternate' 
                        else ColorAssignment.FIXED)
    
    # Create and run the benchmark
    benchmark = NardeBenchmark(
        max_steps_per_game=args.max_steps,
        verbose=args.verbose,
        debug_episodes=args.debug
    )
    
    results = benchmark.run_benchmark(
        agent1, agent2, num_games=args.games, color_assignment=color_assignment
    )
    
    # Print the results
    benchmark.print_benchmark_results(results) 