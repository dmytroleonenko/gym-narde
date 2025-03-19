#!/usr/bin/env python3
"""
Script to train MuZero for Narde and evaluate against RandomAgent.
"""

import os
import argparse
import subprocess
import torch
import time
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from muzero.training import train_muzero


def run_single_evaluation(game_id, model_path, max_steps, num_simulations=50, temp=0.0, seed_offset=1000):
    """
    Run a single evaluation game in a separate process
    
    Args:
        game_id: The ID of the game (used for seeding)
        model_path: Path to the MuZero model
        max_steps: Maximum steps per game
        num_simulations: Number of MCTS simulations per move
        temp: Temperature for action selection
        seed_offset: Offset for random seed to ensure different games
        
    Returns:
        Game result dictionary
    """
    # Set different seeds for each process
    seed = seed_offset + game_id
    
    # Determine which agent plays as white based on game ID (alternate colors)
    agent1_side = "white" if game_id % 2 == 0 else "black"
    
    # Create a temporary file for output with absolute path
    output_file = os.path.abspath(f"game_result_{game_id}_{int(time.time())}.json")
    
    # Construct the command to run evaluate_with_agents.py for a single game
    cmd = [
        "python3", "evaluate_with_agents.py",
        "--agent1", "muzero.agent.MuZeroAgent",
        "--agent2", "narde_agents.RandomAgent",
        f"--agent1_args=model_path={model_path},num_simulations={num_simulations},temperature={temp}",
        f"--agent1side={agent1_side}",  # Explicitly set which side agent1 plays
        "--episodes", "1",
        "--max_steps", str(max_steps),
        "--seed", str(seed),
        "--json_output", output_file
    ]
    
    # Run the evaluation with error checking
    try:
        # Run process with output capture to aid debugging
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if there was an error in the process
        if result.returncode != 0:
            return {
                "error": f"Process error with code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
                "game_id": game_id
            }
        
        # Parse the results from JSON file
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                game_result = json.load(f)
            os.remove(output_file)  # Clean up
            
            # Add the color information to the result for proper aggregation
            if "stats" in game_result:
                game_result["stats"]["muzero_as_white"] = (agent1_side == "white")
                game_result["stats"]["muzero_as_black"] = (agent1_side == "black")
                game_result["stats"]["game_id"] = game_id
                
            return game_result
        else:
            return {
                "error": f"JSON file not created: {output_file}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
                "game_id": game_id
            }
    except Exception as e:
        return {"error": f"Exception in evaluation: {str(e)}", "game_id": game_id}


def main():
    parser = argparse.ArgumentParser(description="Train MuZero for Narde and evaluate against RandomAgent.")
    # Training parameters
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to train for")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Size of the replay buffer")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--simulations", type=int, default=50, help="Number of MCTS simulations per move")
    parser.add_argument("--temp-init", type=float, default=1.0, help="Initial temperature for action selection")
    parser.add_argument("--temp-final", type=float, default=0.1, help="Final temperature for action selection")
    parser.add_argument("--discount", type=float, default=0.997, help="Discount factor for rewards")
    parser.add_argument("--save-interval", type=int, default=50, help="Save the model every save_interval episodes")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Dimensionality of the latent state")
    parser.add_argument("--device", type=str, default="auto", help="Device to run the training on (auto, cpu, cuda, mps)")
    parser.add_argument("--continue-training", action="store_true", help="Continue training from the model specified by --model-path")
    parser.add_argument("--previous-episodes", type=int, default=0, help="Number of episodes previously trained (for temperature annealing)")
    parser.add_argument("--parallel-self-play", "-ps", type=int, default=1, 
                      help="Number of parallel processes for self-play during training")
    
    # Profiling parameters
    parser.add_argument("--enable-profiling", action="store_true", help="Enable PyTorch profiler")
    parser.add_argument("--profile-episodes", type=int, default=5, help="Number of episodes to profile")
    
    # Evaluation parameters
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--eval-max-steps", type=int, default=500, help="Maximum steps per episode during evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only evaluate")
    parser.add_argument("--model-path", type=str, default="muzero/models/muzero_model_final.pth", 
                      help="Path to MuZero model for evaluation or continuing training")
    parser.add_argument("--parallel", "-p", type=int, default=1, 
                      help="Number of parallel processes for evaluation")
    
    args = parser.parse_args()
    
    model_path = args.model_path
    
    # Use a smaller batch size for very small episode counts
    if args.episodes < 5 and args.batch_size > args.episodes * 10:
        # Set batch size to something reasonable for small episode counts
        args.batch_size = max(4, args.episodes * 10)
        print(f"Adjusting batch size to {args.batch_size} for small episode count")
    
    # Train MuZero if not in eval-only mode
    if not args.eval_only:
        print("=== Starting MuZero Training ===")
        
        # If continuing training, load the model first
        start_with_model = None
        if args.continue_training:
            if os.path.exists(args.model_path):
                print(f"Continuing training from {args.model_path}")
                start_with_model = args.model_path
                total_episodes = args.previous_episodes + args.episodes
                print(f"Total episodes (including previous): {total_episodes}")
            else:
                print(f"Warning: Model path {args.model_path} not found. Starting with a new model.")
                total_episodes = args.episodes
        else:
            total_episodes = args.episodes
        
        network = train_muzero(
            num_episodes=args.episodes,
            replay_buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            lr=args.lr,
            num_simulations=args.simulations,
            temperature_init=args.temp_init,
            temperature_final=args.temp_final,
            discount=args.discount,
            save_interval=args.save_interval,
            hidden_dim=args.hidden_dim,
            device_str=args.device,
            enable_profiling=args.enable_profiling,
            profile_episodes=args.profile_episodes,
            start_with_model=start_with_model,
            previous_episodes=args.previous_episodes,
            total_episodes=total_episodes,
            parallel_self_play=args.parallel_self_play
        )
        model_path = "muzero/models/muzero_model_final.pth"
    
    # Evaluate MuZero against RandomAgent
    print(f"\n=== Starting Evaluation: MuZero vs RandomAgent ===")
    
    # Single-process evaluation
    if args.parallel <= 1:
        print("Running evaluation in single process mode")
        # Construct the command to run evaluate_with_agents.py
        cmd = [
            "python3", "evaluate_with_agents.py",
            "--agent1", "muzero.agent.MuZeroAgent",
            "--agent2", "narde_agents.RandomAgent",
            f"--agent1_args=model_path={model_path},num_simulations=50,temperature=0.0",
            "--episodes", str(args.eval_episodes),
            "--max_steps", str(args.eval_max_steps),
            "--verbose"
        ]
        
        # Run the evaluation
        subprocess.run(cmd)
    else:
        # Parallel evaluation mode
        print(f"Running evaluation in parallel mode with {args.parallel} processes")
        
        # Initialize statistics
        results = {
            "muzero_wins": 0,
            "random_wins": 0,
            "draws": 0,
            "total_steps": 0,
            "white_wins": 0,
            "black_wins": 0,
            "muzero_white_wins": 0,
            "muzero_black_wins": 0,
            "muzero_white_games": 0,
            "muzero_black_games": 0,
            "white_checkers_off": 0,
            "black_checkers_off": 0,
            "max_steps": 0
        }
        
        # Set up the partial function for evaluation
        evaluate_game = partial(
            run_single_evaluation,
            model_path=model_path,
            max_steps=args.eval_max_steps,
            num_simulations=50  # Use consistent default of 50 simulations for evaluation
        )
        
        # Progress tracking variables
        num_games = args.eval_episodes
        progress_interval = max(1, num_games // 10)  # Report after every 10% of games
        start_time = time.time()
        
        # Run evaluations in parallel
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # Process games as they complete
            completed_games = 0
            
            for i, result in enumerate(executor.map(evaluate_game, range(num_games))):
                completed_games += 1
                
                # Handle errors
                if "error" in result:
                    print(f"Error in game {i}: {result['error']}")
                    continue
                
                # Extract game statistics
                if "stats" in result:
                    stats = result["stats"]
                    
                    # Update wins/draws counters
                    if stats.get("winner") == "MuZeroAgent1":
                        results["muzero_wins"] += 1
                    elif stats.get("winner") == "RandomAgent2":
                        results["random_wins"] += 1
                    else:
                        results["draws"] += 1
                    
                    # Update steps
                    steps = stats.get("num_steps", 0)
                    results["total_steps"] += steps
                    results["max_steps"] = max(results["max_steps"], steps)
                    
                    # Update color-based statistics using the explicit color assignment information
                    if stats.get("muzero_as_white", False):
                        results["muzero_white_games"] += 1
                        if stats.get("winner") == "MuZeroAgent1":
                            results["white_wins"] += 1
                            results["muzero_white_wins"] += 1
                    elif stats.get("muzero_as_black", False):
                        results["muzero_black_games"] += 1
                        if stats.get("winner") == "MuZeroAgent1":
                            results["black_wins"] += 1
                            results["muzero_black_wins"] += 1
                    
                    # Update checkers borne off
                    results["white_checkers_off"] += stats.get("white_checkers_off", 0)
                    results["black_checkers_off"] += stats.get("black_checkers_off", 0)
                
                # Report progress periodically
                if completed_games % progress_interval == 0 or completed_games == num_games:
                    elapsed = time.time() - start_time
                    games_per_sec = completed_games / elapsed if elapsed > 0 else 0
                    remaining = (num_games - completed_games) / games_per_sec if games_per_sec > 0 else 0
                    
                    muzero_win_rate = (results["muzero_wins"] / completed_games) * 100 if completed_games > 0 else 0
                    random_win_rate = (results["random_wins"] / completed_games) * 100 if completed_games > 0 else 0
                    draw_rate = (results["draws"] / completed_games) * 100 if completed_games > 0 else 0
                    avg_steps = results["total_steps"] / completed_games if completed_games > 0 else 0
                    
                    print(f"Completed {completed_games}/{num_games} games ({games_per_sec:.2f} games/sec)")
                    print(f"  MuZeroAgent1 win rate: {muzero_win_rate:.2f}%")
                    print(f"  RandomAgent2 win rate: {random_win_rate:.2f}%")
                    print(f"  Draws: {draw_rate:.2f}%")
                    print(f"  Average steps: {avg_steps:.2f}")
                    print(f"  Estimated time remaining: {remaining:.1f} seconds")
        
        # Print final benchmark results
        total_time = time.time() - start_time
        print("\nBenchmark complete")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per game: {total_time/num_games:.4f} seconds")
        
        # Calculate final statistics
        muzero_win_rate = results["muzero_wins"] / num_games * 100
        random_win_rate = results["random_wins"] / num_games * 100
        draw_rate = results["draws"] / num_games * 100
        
        white_win_rate = results["white_wins"] / num_games * 100
        black_win_rate = (num_games - results["white_wins"] - results["draws"]) / num_games * 100
        
        muzero_white_win_rate = results["muzero_white_wins"] / results["muzero_white_games"] * 100 if results["muzero_white_games"] > 0 else 0
        muzero_black_win_rate = results["muzero_black_wins"] / results["muzero_black_games"] * 100 if results["muzero_black_games"] > 0 else 0
        
        avg_white_checkers = results["white_checkers_off"] / num_games
        avg_black_checkers = results["black_checkers_off"] / num_games
        avg_steps_per_checker = results["total_steps"] / (results["white_checkers_off"] + results["black_checkers_off"]) if (results["white_checkers_off"] + results["black_checkers_off"]) > 0 else 0
        
        # Print detailed results
        print("\n============================================================")
        print(f"Benchmark Results: MuZeroAgent1 vs RandomAgent2")
        print("============================================================")
        print(f"Total Games: {num_games}")
        print(f"MuZeroAgent1 Win Rate: {muzero_win_rate:.2f}% ({results['muzero_wins']} wins)")
        print(f"RandomAgent2 Win Rate: {random_win_rate:.2f}% ({results['random_wins']} wins)")
        print(f"Draws: {draw_rate:.2f}% ({results['draws']} draws)")
        print()
        print(f"White Color Win Rate: {white_win_rate:.2f}% ({results['white_wins']}/{num_games})")
        print(f"Black Color Win Rate: {black_win_rate:.2f}% ({num_games - results['white_wins'] - results['draws']}/{num_games})")
        print(f"MuZeroAgent1 as White: {muzero_white_win_rate:.2f}% ({results['muzero_white_wins']}/{results['muzero_white_games']})")
        print(f"MuZeroAgent1 as Black: {muzero_black_win_rate:.2f}% ({results['muzero_black_wins']}/{results['muzero_black_games']})")
        print(f"RandomAgent2 as White: {100-muzero_black_win_rate:.2f}% ({results['muzero_black_games']-results['muzero_black_wins']}/{results['muzero_black_games']})")
        print(f"RandomAgent2 as Black: {100-muzero_white_win_rate:.2f}% ({results['muzero_white_games']-results['muzero_white_wins']}/{results['muzero_white_games']})")
        print()
        print(f"Average Steps Per Game: {results['total_steps']/num_games:.2f}")
        print(f"Maximum Steps in a Game: {results['max_steps']}")
        print(f"Average White Checkers Borne Off: {avg_white_checkers:.2f}")
        print(f"Average Black Checkers Borne Off: {avg_black_checkers:.2f}")
        print(f"Average Steps Per Checker Borne Off: {avg_steps_per_checker:.2f}")
        print("============================================================")
    
    print("Training and evaluation complete!")


if __name__ == "__main__":
    main() 