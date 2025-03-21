#!/usr/bin/env python3
"""
Script to evaluate different agents playing the Narde game.
"""

import gym_narde  # Import this first to register the environment
import gymnasium as gym
import argparse
import sys
import importlib
from benchmarks.narde_benchmark import NardeBenchmark, ColorAssignment
import os


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
    # Special case for MuZero agent
    if agent_class_path == "muzero_agents.MuZeroAgent" or agent_class_path.endswith(".MuZeroAgent"):
        from muzero.agent import MuZeroAgent
        from benchmarks.narde_benchmark import GenericNardeAgent
        
        # Create a wrapper that adapts MuZeroAgent to the benchmark interface
        class MuZeroAgentAdapter(GenericNardeAgent):
            def __init__(self, model_path, name="MuZeroAgent", num_simulations=50, temperature=0.0, device="auto"):
                self._name = name
                self._agent = MuZeroAgent(
                    model_path=model_path,
                    num_simulations=num_simulations,
                    temperature=temperature,
                    device=device,
                    name=name
                )
            
            @property
            def name(self):
                return self._name
                
            def reset(self):
                if hasattr(self._agent, 'reset'):
                    self._agent.reset()
                    
            def select_action(self, env, state):
                return self._agent.select_action(env, state)
                
        # Parse additional arguments
        agent_args = {"model_path": model_path, "name": f"MuZeroAgent{name_suffix}"}
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
        
        return MuZeroAgentAdapter(**agent_args)
        
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Narde agents')
    
    # Agent selection
    parser.add_argument('--agent1', type=str, default='narde_agents.RandomAgent',
                      help='First agent class (module.ClassName format)')
    parser.add_argument('--agent2', type=str, default='narde_agents.RandomAgent',
                      help='Second agent class (module.ClassName format)')
    
    # Agent parameters
    parser.add_argument('--model', type=str, default='narde_dqn_model.pth',
                      help='Path to model file (if using DQN agents)')
    parser.add_argument('--agent1_args', type=str, default='',
                      help='Additional arguments for agent1 (comma-separated key=value pairs)')
    parser.add_argument('--agent2_args', type=str, default='',
                      help='Additional arguments for agent2 (comma-separated key=value pairs)')
    
    # Evaluation parameters
    parser.add_argument('--agent1side', type=str, default='alternate',
                      choices=['white', 'black', 'alternate'],
                      help='Side for agent1 to play (white, black, or alternate)')
    parser.add_argument('--episodes', type=int, default=100,
                      help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=500,
                      help='Maximum steps per episode')
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed progress')
    parser.add_argument('--debug', type=int, default=0,
                      help='Number of episodes to show debug info for')
    parser.add_argument('--json_output', type=str, default='',
                      help='Output file for JSON results (for parallel processing)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main function to run the evaluation."""
    args = parse_args()
    
    # Set random seed if specified (for parallel evaluations)
    if args.seed is not None:
        import numpy as np
        import random
        import torch
        
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Create the agents
    agent1 = create_agent(args.agent1, args.agent1_args, args.model, "1")
    agent2 = create_agent(args.agent2, args.agent2_args, args.model, "2")
    
    # Determine color assignment
    if args.agent1side == 'white':
        color_assignment = ColorAssignment.FIXED
        white_agent, black_agent = agent1, agent2
    elif args.agent1side == 'black':
        color_assignment = ColorAssignment.FIXED
        white_agent, black_agent = agent2, agent1
    else:  # alternate
        color_assignment = ColorAssignment.ALTERNATE
        # For alternate, the agents will switch colors in the benchmark
        white_agent, black_agent = None, None
    
    # Create the benchmark
    benchmark = NardeBenchmark(
        max_steps_per_game=args.max_steps,
        verbose=args.verbose,
        debug_episodes=args.debug
    )
    
    # Run the benchmark
    if color_assignment == ColorAssignment.ALTERNATE:
        results = benchmark.run_benchmark(
            agent1, agent2, num_games=args.episodes, color_assignment=color_assignment
        )
    else:
        # For fixed colors, we run with specific white and black agents
        # We need to simulate the same API as run_benchmark but with fixed agents
        results = benchmark.run_benchmark(
            white_agent, black_agent, num_games=args.episodes, color_assignment=color_assignment
        )
    
    # Print the results
    benchmark.print_benchmark_results(results)
    
    # Write JSON output if specified (for parallel processing)
    if args.json_output:
        import json
        
        try:
            # Structure the results for JSON output
            output_data = {"stats": {}}
            
            # For single game mode, extract details from the first game
            if args.episodes == 1 and hasattr(results, 'game_results') and len(results.game_results) == 1:
                game = results.game_results[0]
                
                # Extract winner information
                winner_name = None
                if game.winner == "white" and game.white_agent_name.endswith("1"):
                    winner_name = "MuZeroAgent1"
                elif game.winner == "black" and game.black_agent_name.endswith("1"):
                    winner_name = "MuZeroAgent1" 
                elif game.winner == "white" or game.winner == "black":
                    winner_name = "RandomAgent2"
                
                # Create a JSON-serializable dictionary with key statistics
                output_data["stats"] = {
                    "winner": winner_name,
                    "num_steps": game.steps,
                    "white_player": "MuZeroAgent1" if game.white_agent_name.endswith("1") else "RandomAgent2",
                    "black_player": "MuZeroAgent1" if game.black_agent_name.endswith("1") else "RandomAgent2",
                    "white_checkers_off": game.white_borne_off,
                    "black_checkers_off": game.black_borne_off
                }
            else:
                # For multi-game mode, provide summary statistics
                output_data["stats"] = {
                    "total_games": results.total_games,
                    "agent1_wins": results.agent1_wins,
                    "agent2_wins": results.agent2_wins,
                    "draws": results.draws,
                    "avg_steps": results.avg_steps
                }
            
            # Write to the specified JSON file
            with open(args.json_output, 'w') as f:
                json.dump(output_data, f)
                
            # Verify the file was created for debugging purposes
            if not os.path.exists(args.json_output):
                print(f"Warning: JSON file was not created at {args.json_output}", file=sys.stderr)
            else:
                file_size = os.path.getsize(args.json_output)
                if file_size == 0:
                    print(f"Warning: JSON file at {args.json_output} is empty", file=sys.stderr)
                
        except Exception as e:
            import traceback
            print(f"Error creating JSON output file: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            print(f"Results type: {type(results)}", file=sys.stderr)
            print(f"Results dir: {dir(results)}", file=sys.stderr)


if __name__ == "__main__":
    main()