#!/usr/bin/env python3
"""
Tests for the NardeBenchmark class.
"""

import unittest
import pytest
import random
import numpy as np
from unittest.mock import MagicMock, patch
import os
import sys

# Add the parent directory to the Python path to access the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from narde_benchmark import NardeBenchmark, ColorAssignment, GameResult, BenchmarkResult
from narde_agents import RandomAgent

# Mock environment and game for testing
class MockGame:
    def __init__(self):
        self.board = np.zeros(24)
        self.borne_off_white = 0
        self.borne_off_black = 0
        self.valid_moves = [(0, 1), (5, 6)]
        self.winner = None
    
    def get_valid_moves(self):
        return self.valid_moves
    
    def rotate_board_for_next_player(self):
        # Just a mock implementation that doesn't actually rotate
        pass
    
    def get_winner(self):
        # Return the winner: +1 for White, -1 for Black
        return self.winner

class MockEnv:
    def __init__(self):
        self.game = MockGame()
        self.dice = [1, 2]
        self.observation_space = MagicMock()
        self.action_space = [MagicMock(), MagicMock()]
        # Add unwrapped attribute that returns self
        self.unwrapped = self
    
    def reset(self):
        self.game = MockGame()
        self.dice = [1, 2]
        return np.zeros(24), {}
    
    def step(self, action):
        # Reduce dice on each step
        if len(self.dice) > 0:
            self.dice.pop(0)
        
        # Return mock values
        return np.zeros(24), 0.0, False, False, {}
    
    def _get_obs(self):
        return np.zeros(24)


@pytest.fixture
def mock_env():
    """Create a mock environment for testing."""
    mock_env = MockEnv()
    with patch('gymnasium.make', return_value=mock_env) as mock_make:
        yield mock_make


class TestNardeBenchmark:
    """Tests for the NardeBenchmark class."""
    
    def test_init(self, mock_env):
        """Test benchmark initialization."""
        benchmark = NardeBenchmark(max_steps_per_game=100, verbose=True, debug_episodes=1)
        assert benchmark.max_steps_per_game == 100
        assert benchmark.verbose == True
        assert benchmark.debug_episodes == 1
    
    def test_run_single_game_normal_finish(self, mock_env):
        """Test running a single game with a normal finish (win)."""
        # Create benchmark
        benchmark = NardeBenchmark(max_steps_per_game=100)
        
        # Create agents
        agent1 = RandomAgent(name="White")
        agent2 = RandomAgent(name="Black")
        
        # Configure mock environment to end the game with White as winner
        benchmark.env_unwrapped.game.winner = 1  # White wins
        benchmark.env_unwrapped.step = lambda action: (np.zeros(24), 0.0, True, False, {})  # Game ends after first step
        
        # Run a single game
        result = benchmark.run_single_game(agent1, agent2)
        
        # Check the result
        assert result.white_agent_name == "White"
        assert result.black_agent_name == "Black"
        assert result.winner == "white"  # Should be "white" since we set winner to 1
        assert result.steps > 0
        assert not result.truncated
    
    def test_run_single_game_truncated(self, mock_env):
        """Test running a single game that gets truncated."""
        # Create benchmark with small max steps
        benchmark = NardeBenchmark(max_steps_per_game=2)
        
        # Create agents
        agent1 = RandomAgent(name="White")
        agent2 = RandomAgent(name="Black")
        
        # Configure mock environment to never end
        benchmark.env_unwrapped.step = lambda action: (np.zeros(24), 0.0, False, False, {})
        
        # Run a single game that will be truncated after max_steps
        result = benchmark.run_single_game(agent1, agent2)
        
        # Check the result
        assert result.white_agent_name == "White"
        assert result.black_agent_name == "Black"
        assert result.winner is None  # Game was truncated, no winner
        assert result.steps == 2  # Should match max_steps
        assert result.truncated
    
    def test_run_benchmark_alternating(self, mock_env):
        """Test running multiple games with alternating colors."""
        # Create benchmark
        benchmark = NardeBenchmark(max_steps_per_game=10)
        
        # Create agents
        agent1 = RandomAgent(name="Agent1")
        agent2 = RandomAgent(name="Agent2")
        
        # Keep track of the current game and who's playing as white
        game_count = [0]
        
        # Mock run_single_game to control the winner
        def mock_run_single_game(white_agent, black_agent, episode=0):
            game_count[0] += 1
            is_white_winner = game_count[0] % 2 == 0
            
            # Based on game number, alternate who wins
            # Games 1 and 3: Black wins, Games 2 and 4: White wins
            if is_white_winner:
                winner = "white"
            else:
                winner = "black"
            
            # In games 0 and 2, agent1 is white
            # In games 1 and 3, agent2 is white
            white_is_agent1 = game_count[0] % 2 == 1
            
            # Determine which agent wins based on the color and winner
            agent1_wins = (white_is_agent1 and winner == "white") or (not white_is_agent1 and winner == "black")
            
            # Create a GameResult with the appropriate winner
            return GameResult(
                white_agent_name=white_agent.name,
                black_agent_name=black_agent.name,
                winner=winner,
                steps=1,
                white_borne_off=0,
                black_borne_off=0,
                truncated=False
            )
        
        # Replace the run_single_game method
        benchmark.run_single_game = mock_run_single_game
        
        # Run benchmark with alternating colors
        results = benchmark.run_benchmark(
            agent1, agent2, num_games=4, color_assignment=ColorAssignment.ALTERNATE
        )
        
        # Check the results
        assert results.total_games == 4
        assert results.agent1_wins == 2  # Agent1 should win games 1 and 3
        assert results.agent2_wins == 2  # Agent2 should win games 2 and 4
        assert results.agent1_as_white_games == 2
        assert results.agent1_as_black_games == 2
    
    def test_run_benchmark_fixed_colors(self, mock_env):
        """Test running multiple games with fixed colors."""
        # Create benchmark
        benchmark = NardeBenchmark(max_steps_per_game=10)
        
        # Create agents
        agent1 = RandomAgent(name="Agent1")
        agent2 = RandomAgent(name="Agent2")
        
        # Mock run_single_game to have White always win
        def mock_run_single_game(white_agent, black_agent, episode=0):
            return GameResult(
                white_agent_name=white_agent.name,
                black_agent_name=black_agent.name,
                winner="white",  # White always wins
                steps=1,
                white_borne_off=0,
                black_borne_off=0,
                truncated=False
            )
        
        # Replace the run_single_game method
        benchmark.run_single_game = mock_run_single_game
        
        # Run benchmark with fixed colors (agent1 always White)
        results = benchmark.run_benchmark(
            agent1, agent2, num_games=5, color_assignment=ColorAssignment.FIXED
        )
        
        # Check the results
        assert results.total_games == 5
        assert results.agent1_wins == 5  # Agent1 should win all games since it's always White
        assert results.agent2_wins == 0
        assert results.agent1_as_white_games == 5
        assert results.agent1_as_black_games == 0
    
    def test_benchmark_result_properties(self):
        """Test the properties on the BenchmarkResult class."""
        # Create a result with some stats
        result = BenchmarkResult(
            agent1_name="Agent1",
            agent2_name="Agent2",
            total_games=10,
            agent1_wins=6,
            agent2_wins=3,
            draws=1,
            agent1_as_white_wins=4,
            agent1_as_white_games=5,
            agent1_as_black_wins=2,
            agent1_as_black_games=5,
            avg_steps=25.5,
            max_steps=40,
            avg_white_borne_off=12.5,
            avg_black_borne_off=7.2,
            game_results=[]
        )
        
        # Check property calculations
        assert result.agent1_win_rate == 0.6
        assert result.agent2_win_rate == 0.3
        assert result.draw_rate == 0.1
        assert result.agent1_as_white_win_rate == 0.8
        assert result.agent1_as_black_win_rate == 0.4
    
    def test_print_benchmark_results(self, mock_env, capsys):
        """Test that printing benchmark results works correctly."""
        # Create benchmark
        benchmark = NardeBenchmark()
        
        # Create a result with some stats
        result = BenchmarkResult(
            agent1_name="Agent1",
            agent2_name="Agent2",
            total_games=10,
            agent1_wins=6,
            agent2_wins=3,
            draws=1,
            agent1_as_white_wins=4,
            agent1_as_white_games=5,
            agent1_as_black_wins=2,
            agent1_as_black_games=5,
            avg_steps=25.5,
            max_steps=40,
            avg_white_borne_off=12.5,
            avg_black_borne_off=7.2,
            game_results=[]
        )
        
        # Print the results
        benchmark.print_benchmark_results(result)
        
        # Check that the output contains key information
        captured = capsys.readouterr()
        assert "Benchmark Results: Agent1 vs Agent2" in captured.out
        assert "Total Games: 10" in captured.out
        assert "Agent1 Win Rate: 60.00%" in captured.out
        assert "Agent2 Win Rate: 30.00%" in captured.out
        assert "Draws: 10.00%" in captured.out
        assert "Agent1 as White: 80.00%" in captured.out
        assert "Agent1 as Black: 40.00%" in captured.out


class TestBenchmarkResult:
    """Tests for the BenchmarkResult class and its properties."""
    
    def test_win_rate_calculations(self):
        """Test the win rate calculations in BenchmarkResult."""
        # Create a result with some stats
        result = BenchmarkResult(
            agent1_name="Agent1",
            agent2_name="Agent2",
            total_games=10,
            agent1_wins=6,
            agent2_wins=3,
            draws=1,
            agent1_as_white_wins=4,
            agent1_as_white_games=5,
            agent1_as_black_wins=2,
            agent1_as_black_games=5,
            avg_steps=25.5,
            max_steps=40,
            avg_white_borne_off=12.5,
            avg_black_borne_off=7.2,
            game_results=[]
        )
        
        # Check property calculations
        assert result.agent1_win_rate == 0.6
        assert result.agent2_win_rate == 0.3
        assert result.draw_rate == 0.1
        assert result.agent1_as_white_win_rate == 0.8
        assert result.agent1_as_black_win_rate == 0.4
    
    def test_zero_division_handling(self):
        """Test that win rate calculations handle zero division properly."""
        # Create a result with zero games
        result = BenchmarkResult(
            agent1_name="Agent1",
            agent2_name="Agent2",
            total_games=0,
            agent1_wins=0,
            agent2_wins=0,
            draws=0,
            agent1_as_white_wins=0,
            agent1_as_white_games=0,
            agent1_as_black_wins=0,
            agent1_as_black_games=0,
            avg_steps=0,
            max_steps=0,
            avg_white_borne_off=0,
            avg_black_borne_off=0,
            game_results=[]
        )
        
        # Check that zero division is handled
        assert result.agent1_win_rate == 0
        assert result.agent2_win_rate == 0
        assert result.draw_rate == 0
        assert result.agent1_as_white_win_rate == 0
        assert result.agent1_as_black_win_rate == 0


class TestGameResult:
    """Tests for the GameResult class."""
    
    def test_game_result_creation(self):
        """Test creating GameResult objects."""
        # Create a game result with White as winner
        white_win = GameResult(
            white_agent_name="Agent1",
            black_agent_name="Agent2",
            winner="white",
            steps=30,
            white_borne_off=15,
            black_borne_off=10,
            truncated=False
        )
        
        assert white_win.white_agent_name == "Agent1"
        assert white_win.black_agent_name == "Agent2"
        assert white_win.winner == "white"
        assert white_win.steps == 30
        assert white_win.white_borne_off == 15
        assert white_win.black_borne_off == 10
        assert not white_win.truncated
        
        # Create a game result with Black as winner
        black_win = GameResult(
            white_agent_name="Agent1",
            black_agent_name="Agent2",
            winner="black",
            steps=25,
            white_borne_off=10,
            black_borne_off=15,
            truncated=False
        )
        
        assert black_win.winner == "black"
        
        # Create a draw result
        draw = GameResult(
            white_agent_name="Agent1",
            black_agent_name="Agent2",
            winner=None,
            steps=100,
            white_borne_off=5,
            black_borne_off=5,
            truncated=True
        )
        
        assert draw.winner is None
        assert draw.truncated


class TestResultTracking:
    """Test the result tracking logic that would be used in the benchmark."""
    
    def test_tracking_alternating_colors(self):
        """Test the logic for tracking results with alternating colors."""
        # Create some game results
        game_results = [
            # Game 1: Agent1 as White, Agent2 as Black, White wins
            GameResult("Agent1", "Agent2", "white", 10, 15, 5, False),
            # Game 2: Agent2 as White, Agent1 as Black, Black wins
            GameResult("Agent2", "Agent1", "black", 15, 5, 15, False),
            # Game 3: Agent1 as White, Agent2 as Black, Black wins
            GameResult("Agent1", "Agent2", "black", 20, 10, 15, False),
            # Game 4: Agent2 as White, Agent1 as Black, White wins
            GameResult("Agent2", "Agent1", "white", 25, 15, 10, False),
        ]
        
        # Initialize tracking metrics
        agent1_wins = 0
        agent2_wins = 0
        draws = 0
        
        agent1_as_white_games = 0
        agent1_as_white_wins = 0
        agent1_as_black_games = 0
        agent1_as_black_wins = 0
        
        # Process each game result
        for i, result in enumerate(game_results):
            # Track color assignment for Agent1
            if result.white_agent_name == "Agent1":
                agent1_as_white_games += 1
                if result.winner == "white":
                    agent1_as_white_wins += 1
            else:
                agent1_as_black_games += 1
                if result.winner == "black":
                    agent1_as_black_wins += 1
            
            # Track overall wins
            if result.winner == "white":
                if result.white_agent_name == "Agent1":
                    agent1_wins += 1
                else:
                    agent2_wins += 1
            elif result.winner == "black":
                if result.black_agent_name == "Agent1":
                    agent1_wins += 1
                else:
                    agent2_wins += 1
            else:
                draws += 1
        
        # Verify tracking metrics
        assert agent1_wins == 2  # Agent1 won games 1 and 2
        assert agent2_wins == 2  # Agent2 won games 3 and 4
        assert draws == 0
        
        assert agent1_as_white_games == 2
        assert agent1_as_white_wins == 1  # Agent1 won 1 game as White
        assert agent1_as_black_games == 2
        assert agent1_as_black_wins == 1  # Agent1 won 1 game as Black
        
        # Create a BenchmarkResult from the tracking metrics
        result = BenchmarkResult(
            agent1_name="Agent1",
            agent2_name="Agent2",
            total_games=len(game_results),
            agent1_wins=agent1_wins,
            agent2_wins=agent2_wins,
            draws=draws,
            agent1_as_white_wins=agent1_as_white_wins,
            agent1_as_white_games=agent1_as_white_games,
            agent1_as_black_wins=agent1_as_black_wins,
            agent1_as_black_games=agent1_as_black_games,
            avg_steps=sum(r.steps for r in game_results) / len(game_results),
            max_steps=max(r.steps for r in game_results),
            avg_white_borne_off=sum(r.white_borne_off for r in game_results) / len(game_results),
            avg_black_borne_off=sum(r.black_borne_off for r in game_results) / len(game_results),
            game_results=game_results
        )
        
        # Check the result
        assert result.agent1_win_rate == 0.5
        assert result.agent2_win_rate == 0.5
        assert result.agent1_as_white_win_rate == 0.5
        assert result.agent1_as_black_win_rate == 0.5
    
    def test_tracking_fixed_colors(self):
        """Test the logic for tracking results with fixed colors."""
        # Create some game results where Agent1 is always White
        game_results = [
            # Game 1: White wins
            GameResult("Agent1", "Agent2", "white", 10, 15, 5, False),
            # Game 2: White wins
            GameResult("Agent1", "Agent2", "white", 15, 15, 10, False),
            # Game 3: Black wins
            GameResult("Agent1", "Agent2", "black", 20, 10, 15, False),
        ]
        
        # Initialize tracking metrics
        agent1_wins = 0
        agent2_wins = 0
        draws = 0
        
        agent1_as_white_games = len(game_results)
        agent1_as_white_wins = 0
        agent1_as_black_games = 0
        agent1_as_black_wins = 0
        
        # Process each game result
        for result in game_results:
            # Agent1 is always White in this test
            if result.winner == "white":
                agent1_wins += 1
                agent1_as_white_wins += 1
            elif result.winner == "black":
                agent2_wins += 1
            else:
                draws += 1
        
        # Verify tracking metrics
        assert agent1_wins == 2  # Agent1 won games 1 and 2
        assert agent2_wins == 1  # Agent2 won game 3
        assert draws == 0
        
        assert agent1_as_white_games == 3
        assert agent1_as_white_wins == 2
        assert agent1_as_black_games == 0
        assert agent1_as_black_wins == 0
        
        # Create a BenchmarkResult from the tracking metrics
        result = BenchmarkResult(
            agent1_name="Agent1",
            agent2_name="Agent2",
            total_games=len(game_results),
            agent1_wins=agent1_wins,
            agent2_wins=agent2_wins,
            draws=draws,
            agent1_as_white_wins=agent1_as_white_wins,
            agent1_as_white_games=agent1_as_white_games,
            agent1_as_black_wins=agent1_as_black_wins,
            agent1_as_black_games=agent1_as_black_games,
            avg_steps=sum(r.steps for r in game_results) / len(game_results),
            max_steps=max(r.steps for r in game_results),
            avg_white_borne_off=sum(r.white_borne_off for r in game_results) / len(game_results),
            avg_black_borne_off=sum(r.black_borne_off for r in game_results) / len(game_results),
            game_results=game_results
        )
        
        # Check the result
        assert result.agent1_win_rate == 2/3
        assert result.agent2_win_rate == 1/3
        assert result.agent1_as_white_win_rate == 2/3
        assert result.agent1_as_black_win_rate == 0 