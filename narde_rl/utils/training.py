"""
Training utilities for reinforcement learning agents.
"""

import os
import time
import numpy as np
import multiprocessing as mp
from datetime import timedelta
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from concurrent.futures import ProcessPoolExecutor

# Define functions for environment operations in separate processes
def env_reset(args):
    """Initialize and reset an environment, returning its initial state"""
    env_id, env_name = args
    
    # Explicitly import gym_narde to ensure environment registration
    import gym_narde
    import gymnasium as gym
    
    env = gym.make(env_name)
    state, info = env.reset()
    env.close()
    return (state, info, env_id)

def env_step(args):
    """Worker function that steps an environment with an action."""
    # Unpack arguments
    env_id, action, env_name, saved_dice = args
    
    # Explicitly import gym_narde to ensure environment registration
    import gym_narde
    import gymnasium as gym
    import numpy as np
    import multiprocessing as mp
    
    # Create environment for this specific step
    env = gym.make(env_name)
    env.reset()
    
    # Set the dice from the saved state (important for move validation)
    if saved_dice is not None:
        env.unwrapped.dice = saved_dice.copy() if isinstance(saved_dice, list) else [saved_dice]
        env.unwrapped.game.dice = saved_dice.copy() if isinstance(saved_dice, list) else [saved_dice]
        if hasattr(env.unwrapped, 'debug') and env.unwrapped.debug and mp.current_process().name == 'MainProcess':
            print(f"DEBUG: Set dice for env {env_id} to {saved_dice}")
    
    # Ensure action is in the correct Gymnasium format
    try:
        # Check if action is already in Gymnasium format
        if isinstance(action, tuple) and len(action) == 2:
            from_pos, to_pos = action
            
            # Only convert if it looks like a move tuple (from_pos, to_pos)
            if (isinstance(from_pos, (int, np.integer)) and 
                (isinstance(to_pos, (int, np.integer)) or to_pos == 'off')):
                
                # Convert numpy types to Python integers if necessary
                if isinstance(from_pos, np.integer):
                    from_pos = int(from_pos)
                if isinstance(to_pos, np.integer) and to_pos != 'off':
                    to_pos = int(to_pos)
                
                # Log the conversion for debugging
                if hasattr(env.unwrapped, 'debug') and env.unwrapped.debug and mp.current_process().name == 'MainProcess':
                    print(f"DEBUG env_step: Converting action ({from_pos}, {to_pos}) for env {env_id}")
                
                # Verify if this move is in valid moves given current dice
                valid_moves = env.unwrapped.game.get_valid_moves(env.unwrapped.dice)
                converted_valid_moves = [(int(f), int(t) if t != 'off' else t) for f, t in valid_moves]
                move_is_valid = (from_pos, to_pos) in converted_valid_moves
                
                if hasattr(env.unwrapped, 'debug') and env.unwrapped.debug and mp.current_process().name == 'MainProcess':
                    print(f"DEBUG: Move ({from_pos}, {to_pos}) valid: {move_is_valid}, dice: {env.unwrapped.dice}")
                    if not move_is_valid:
                        print(f"DEBUG: Valid moves are: {converted_valid_moves}")
                
                # Convert to gym action format: (move_index, move_type)
                if to_pos == 'off':
                    move_index = from_pos * 24
                    move_type = 1  # Bearing off
                else:
                    move_index = from_pos * 24 + to_pos
                    move_type = 0  # Regular move
                
                # Use converted action - ensure they're Python integers
                gym_action = (int(move_index), int(move_type))
            else:
                gym_action = action
        else:
            # Not a tuple, use as is (dummy action or integer)
            # For dummy actions (like 0), create a default invalid action
            if action == 0:
                gym_action = (0, 0)
            else:
                gym_action = action
            
        # Step the environment with the properly formatted action
        next_state, reward, terminated, truncated, info = env.step(gym_action)
        env.close()
        
        return (next_state, reward, terminated, truncated, info, env_id)
    except Exception as e:
        return (None, -1.0, True, False, {"error": str(e), "invalid_move": True, "action": action}, env_id)

def env_get_valid_moves(args):
    """Worker function that gets valid moves for a given dice roll."""
    # Unpack arguments
    env_id, dice, env_name = args
    if dice is None:
        # If no dice provided, create a dummy value
        dice = [1, 1]
    
    # Explicitly import gym_narde to ensure environment registration
    import gym_narde
    import gymnasium as gym
    import multiprocessing as mp
    import numpy as np
    
    # Create environment for this specific operation
    env = gym.make(env_name)
    env.reset()
    
    # Set the environment's dice to match our record
    try:
        # This ensures the environment's dice match what we think they are
        env.unwrapped.dice = dice.copy() if isinstance(dice, list) else dice
        env.unwrapped.game.dice = dice.copy() if isinstance(dice, list) else dice
        
        # Get valid moves
        valid_moves = env.unwrapped.game.get_valid_moves(dice)
        
        # Convert NumPy integers to Python integers for better compatibility
        valid_moves_converted = []
        for move in valid_moves:
            from_pos, to_pos = move
            if isinstance(from_pos, np.integer):
                from_pos = int(from_pos)
            if isinstance(to_pos, np.integer) and to_pos != 'off':
                to_pos = int(to_pos)
            valid_moves_converted.append((from_pos, to_pos))
            
        if hasattr(env.unwrapped, 'debug') and env.unwrapped.debug and mp.current_process().name == 'MainProcess':
            print(f"DEBUG env_get_valid_moves: Env {env_id} with dice {dice} has {len(valid_moves_converted)} valid moves")
            
        env.close()
        return (env_id, valid_moves_converted)
    except Exception as e:
        if hasattr(env.unwrapped, 'debug') and env.unwrapped.debug and mp.current_process().name == 'MainProcess':
            print(f"ERROR in env_get_valid_moves: {e}")
        env.close()
        return (env_id, [])

# Define worker process function
def worker_process(env_id, input_queue, output_queue):
    """
    Worker process that manages an environment instance.
    
    Args:
        env_id: ID for this environment
        input_queue: Queue to receive commands
        output_queue: Queue to send back results
    """
    # Create and initialize environment
    env = None
    
    try:
        # Initialize the environment
        env = gym.make('Narde-v0')
        obs, info = env.reset()
        
        # Main loop - process commands until receiving 'close'
        while True:
            try:
                # Get command from input queue
                cmd, *args = input_queue.get()
                
                # Process command
                if cmd == "reset":
                    obs, info = env.reset()
                    # Send back the observation
                    output_queue.put(("reset_result", obs, info))
                
                elif cmd == "step":
                    action = args[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    # Send back the results
                    output_queue.put(("step_result", obs, reward, terminated, truncated, info))
                
                elif cmd == "get_valid_moves":
                    # Access dice directly from the game
                    dice = env.unwrapped.game.dice
                    
                    # Get valid moves with the current dice
                    valid_moves = env.unwrapped.game.get_valid_moves(dice)
                    
                    # Send back the valid moves with the dice
                    output_queue.put(("valid_moves_result", (valid_moves, dice)))
                
                elif cmd == "close":
                    # Clean up and exit
                    if env:
                        env.close()
                    output_queue.put(("closed",))
                    break
                
                else:
                    # Unknown command
                    output_queue.put(("error", f"Unknown command: {cmd}"))
                    
            except Exception as e:
                # Report any errors but continue processing commands
                output_queue.put(("error", f"Error processing command {cmd}: {str(e)}"))
                
    except Exception as e:
        # Report critical initialization errors
        output_queue.put(("critical_error", f"Worker {env_id} failed: {str(e)}"))
    
    finally:
        # Ensure environment is closed
        if env:
            try:
                env.close()
            except:
                pass

class ParallelEnv:
    """
    A class that runs multiple gym-narde environments in parallel using multiprocessing.
    """
    
    def __init__(self, env_name, num_envs=4, max_workers=None):
        self.env_name = env_name
        self.num_envs = num_envs
        self.max_workers = max_workers if max_workers is not None else min(num_envs, mp.cpu_count())
        
        # Store the most recent states and info dictionaries
        self.states = [None] * num_envs
        self.latest_infos = [{}] * num_envs
        self.dice_list = [None] * num_envs  # Store dice for each environment
        
        # Initialize environments in worker processes
        with mp.Pool(processes=self.max_workers) as pool:
            results = pool.map(env_reset, [(i, self.env_name) for i in range(num_envs)])
            
        for i, (state, info, env_id) in enumerate(results):
            self.states[env_id] = state
            self.latest_infos[env_id] = info
            if 'dice' in info:
                self.dice_list[env_id] = info['dice']
    
    def reset(self):
        """Reset all parallel environments."""
        # Reset dice list for all environments
        self.dice_list = [None] * self.num_envs
        
        # Create args with appropriate environment IDs
        args = [(i, self.env_name) for i in range(self.num_envs)]
        
        with mp.Pool(processes=self.max_workers) as pool:
            results = pool.map(env_reset, args)
            
        # Process results and update state
        states = [None] * self.num_envs
        self.latest_infos = [{} for _ in range(self.num_envs)]
        
        for state, info, env_id in results:
            states[env_id] = state
            self.states[env_id] = state
            self.latest_infos[env_id] = info
            
            # Store dice information if available
            if 'dice' in info:
                self.dice_list[env_id] = info['dice']
            
        return states
    
    def step(self, actions):
        """Take steps in all parallel environments with the given actions."""
        args = []
        for i in range(len(actions)):
            # Skip actions for done environments
            if i >= len(self.dice_list) or self.dice_list[i] is None:
                # This environment is not properly initialized or is already done
                continue
                
            args.append((i, actions[i], self.env_name, self.dice_list[i]))
        
        if not args:
            # No valid environments to step
            return self.states, [0] * len(self.states), [True] * len(self.states), [False] * len(self.states), [{} for _ in range(len(self.states))]
        
        with mp.Pool(processes=self.max_workers) as pool:
            results = pool.map(env_step, args)
        
        next_states = list(self.states)  # Make a copy to preserve previous states
        rewards = [0.0] * len(self.states)
        dones = [True] * len(self.states)  # Assume all done unless updated
        truncateds = [False] * len(self.states)
        infos = [{} for _ in range(len(self.states))]
        
        for next_state, reward, done, truncated, info, env_id in results:
            # Update states for this environment
            next_states[env_id] = next_state
            rewards[env_id] = reward
            dones[env_id] = done
            truncateds[env_id] = truncated
            infos[env_id] = info
            
            # Store state for this environment
            self.states[env_id] = next_state
            self.latest_infos[env_id] = info
            
            # Update dice information from info dictionary
            if 'dice' in info:
                self.dice_list[env_id] = info['dice']
            elif done:
                # If environment is done, clear its dice
                self.dice_list[env_id] = None
        
        return next_states, rewards, dones, truncateds, infos
    
    def get_valid_moves(self, env_indices=None):
        """Get valid moves for the specified environments."""
        if env_indices is None:
            env_indices = range(self.num_envs)
        
        args = [(i, self.dice_list[i], self.env_name) for i in env_indices]
        
        with mp.Pool(processes=self.max_workers) as pool:
            valid_moves_result = pool.map(env_get_valid_moves, args)
        
        # Process results
        all_valid_moves = {}
        for env_id, moves in valid_moves_result:
            all_valid_moves[env_id] = moves
        
        return all_valid_moves
        
    def close(self):
        """Close all environments."""
        # Nothing to do here as we create environments on demand in workers
        pass

def train(agent, env, episodes=1000, max_steps=500, epsilon_decay=0.995, 
          debug=False, verbose=False):
    """
    Train the DQN agent on the Narde environment.
    
    Args:
        agent: The DQN agent
        env: The environment
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        epsilon_decay: Rate at which to decay exploration
        debug: Whether to run in debug mode
        verbose: Whether to print verbose information
    
    Returns:
        List of rewards for each episode
    """
    # Initialize training metrics
    rewards_history = []
    win_count = 0
    loss_count = 0
    draw_count = 0
    steps_history = []
    start_time = time.time()
    summary_interval = 5  # Display summary every 5 episodes for short runs
    
    if episodes > 100:
        summary_interval = 100  # Use longer interval for longer training runs
    
    print(f"Starting training for {episodes} episodes with max {max_steps} steps per episode")
    
    # Progress tracking with tqdm
    pbar = tqdm(range(episodes), desc="Training Progress")
    
    # Training loop
    for episode in pbar:
        # Reset environment at the start of each episode
        states = env.reset()
        
        episode_rewards = [0] * len(states)
        done = [False] * len(states)
        step_count = 0
        
        # Step through the episode
        for step in range(max_steps):
            # For each environment, get action from agent
            actions = []
            valid_moves_list = env.get_valid_moves()
            
            for i, state in enumerate(states):
                if not done[i]:
                    # Get valid moves for this environment
                    valid_moves = valid_moves_list.get(i, [])
                    
                    # Debug: Print valid moves information
                    if debug and episode < 2:  # Only print for first few episodes
                        print(f"\nEpisode {episode}, Step {step}, Env {i}")
                        print(f"Valid moves count: {len(valid_moves)}")
                        if len(valid_moves) > 0:
                            print(f"Sample valid moves: {valid_moves[:3]}")
                    
                    # If no valid moves, skip
                    if not valid_moves:
                        actions.append(0)  # Dummy action
                        if debug and episode < 2:
                            print(f"No valid moves available for env {i}")
                        continue
                    
                    # Get and take action
                    action = agent.act(state, valid_moves=valid_moves, training=True, debug=debug)
                    
                    # Debug: Print action information
                    if debug and episode < 2:
                        print(f"Agent selected action: {action}")
                        print(f"Action is in valid_moves: {action in valid_moves}")
                        
                    actions.append(action)
                else:
                    # Dummy action for done environments
                    actions.append(0)
            
            # Execute actions in parallel
            next_states, rewards, dones, truncateds, infos = env.step(actions)
            
            # Debug: Print step results
            if debug and episode < 2:
                for i in range(len(rewards)):
                    print(f"Env {i}: Reward={rewards[i]}, Done={dones[i]}, Truncated={truncateds[i]}")
                    if isinstance(infos[i], dict):
                        if 'invalid_move' in infos[i]:
                            print(f"INVALID MOVE detected in env {i}!")
                        if 'dice' in infos[i] and debug:
                            print(f"Dice after move: {infos[i]['dice']}")
                    else:
                        print(f"Info for env {i}: {infos[i]}")
            
            # Combine terminated and truncated flags
            next_done = [term or trunc for term, trunc in zip(dones, truncateds)]
            
            # Update agent for each environment
            for i in range(len(states)):
                if not done[i]:  # Only update for environments that weren't done
                    valid_moves = valid_moves_list.get(i, [])
                    agent.remember(states[i], actions[i], rewards[i], next_states[i], 
                                next_done[i])
                    episode_rewards[i] += rewards[i]
                    
                    # Count wins/losses
                    if next_done[i]:
                        if rewards[i] > 0:
                            win_count += 1
                        elif rewards[i] < 0:
                            loss_count += 1
                        else:
                            draw_count += 1
            
            # Update states and done flags
            states = next_states
            done = next_done
            step_count += 1
            
            # Break if all environments are done
            if all(done):
                break
        
        # Train the agent after each episode
        loss = agent.replay()
        
        # Decay exploration rate
        agent.epsilon *= epsilon_decay
        agent.epsilon = max(agent.epsilon, agent.epsilon_min)
        
        # Record average reward across all environments
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        rewards_history.append(avg_reward)
        steps_history.append(step_count)
        
        # Update progress bar with ETA
        elapsed_time = time.time() - start_time
        remaining_episodes = episodes - episode - 1
        episodes_per_second = (episode + 1) / elapsed_time if elapsed_time > 0 else 0
        remaining_seconds = remaining_episodes / episodes_per_second if episodes_per_second > 0 else 0
        eta = str(timedelta(seconds=int(remaining_seconds)))
        
        # Update progress bar
        pbar.set_description(f"Training Progress (ETA: {eta})")
        
        # Display training summary at intervals or at the end
        if (episode + 1) % summary_interval == 0 or episode == episodes - 1:
            last_n_rewards = rewards_history[-min(summary_interval, len(rewards_history)):]
            last_n_steps = steps_history[-min(summary_interval, len(steps_history)):]
            avg_reward = sum(last_n_rewards) / len(last_n_rewards)
            avg_steps = sum(last_n_steps) / len(last_n_steps)
            win_rate = (win_count / (episode + 1)) * 100 if episode > 0 else 0
            
            # Print summary - use sys.stdout to ensure it's displayed
            import sys
            sys.stdout.write("\n" + "=" * 80 + "\n")
            sys.stdout.write(f"TRAINING SUMMARY - EPISODE {episode+1}/{episodes}\n")
            sys.stdout.write("=" * 80 + "\n")
            sys.stdout.write(f"Last {min(summary_interval, episode+1)} Episodes:\n")
            sys.stdout.write(f"  Average Reward: {avg_reward:.2f}\n")
            sys.stdout.write(f"  Average Steps: {avg_steps:.1f}\n")
            sys.stdout.write(f"  Win Rate: {win_rate:.2f}%\n")
            sys.stdout.write(f"  Current Epsilon: {agent.epsilon:.4f}\n")
            sys.stdout.write("Overall Statistics:\n")
            sys.stdout.write(f"  Wins: {win_count} ({win_count/(episode+1)*100:.2f}%)\n")
            sys.stdout.write(f"  Losses: {loss_count} ({loss_count/(episode+1)*100:.2f}%)\n")
            sys.stdout.write(f"  Draws: {draw_count} ({draw_count/(episode+1)*100:.2f}%)\n")
            sys.stdout.write(f"  Elapsed Time: {str(timedelta(seconds=int(elapsed_time)))}\n")
            sys.stdout.write(f"  Estimated Time Remaining: {eta}\n")
            sys.stdout.write("=" * 80 + "\n")
            sys.stdout.flush()
            
            # Save checkpoint
            if (episode + 1) % (summary_interval * 10) == 0:
                checkpoint_path = f'saved_models/dqn_narde_checkpoint_{episode+1}.pt'
                torch.save(agent.model.network.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
    
    return rewards_history

def test_action_conversion():
    """Test action conversion between (from_pos, to_pos) format and Gymnasium action format."""
    # Create a fresh environment
    import gym_narde
    import gymnasium as gym
    import numpy as np
    
    env = gym.make('Narde-v0')
    unwrapped_env = env.unwrapped
    state, info = env.reset()
    
    # Get valid moves from the game directly
    dice = unwrapped_env.dice
    valid_moves = unwrapped_env.game.get_valid_moves(dice)
    
    if not valid_moves:
        print("No valid moves available.")
        return
    
    # Test the first valid move
    move = valid_moves[0]
    from_pos, to_pos = move
    
    # Convert move to Gymnasium action format
    if to_pos == 'off':
        move_index = from_pos * 24
        move_type = 1  # Bearing off
    else:
        move_index = from_pos * 24 + to_pos
        move_type = 0  # Regular move
    
    gym_action = (move_index, move_type)
    
    # Verify the move can be properly decoded
    decoded_move = unwrapped_env._decode_action(gym_action)
    
    print(f"Original move: {move}")
    print(f"Gym action format: {gym_action}")
    print(f"Decoded move: {decoded_move}")
    print(f"Conversion successful: {decoded_move == move}")
    
    # Test with worker function
    env_step_result = env_step((0, move, 'Narde-v0', None))
    next_state, reward, term, trunc, info, env_id = env_step_result
    
    print(f"Worker function test - Reward: {reward}, Terminated: {term}")
    
    env.close() 