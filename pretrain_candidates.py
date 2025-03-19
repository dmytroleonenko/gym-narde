import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
import copy
import multiprocessing
from collections import deque
import gym_narde  # Import to register the environment

# Define the policy network architecture
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                  "cuda" if torch.cuda.is_available() else "cpu")
        
        # Network architecture
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Move to device
        self.to(self.device)
        
    def forward(self, state):
        """Forward pass through the network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Output action probabilities
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs
    
    def get_action_probs(self, state):
        """Get action probabilities for a state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.forward(state_tensor)
        return action_probs.cpu().numpy()[0]

class REINFORCEAgent:
    def __init__(self, state_size=28, action_size=576, learning_rate=1e-4, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.learning_rate = learning_rate
        
        # Initialize policy network
        self.policy = PolicyNetwork(state_size, action_size)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Episode history
        self.states = []
        self.actions = []
        self.rewards = []
        
    def act(self, state, valid_moves=None, env=None, dice=None, current_player=None):
        """Select an action based on the policy network."""
        # Get action probabilities
        action_probs = self.policy.get_action_probs(state)
        
        # If no valid moves, return skip action
        if not valid_moves or len(valid_moves) == 0:
            return (0, 0)
        
        # Convert valid moves to action indices
        valid_action_indices = []
        for move in valid_moves:
            from_pos, to_pos = move
            
            # Convert to action format
            if to_pos == 'off':
                to_pos = 0  # Use 0 to represent bearing off
                move_type = 1  # Bearing off move type
            else:
                move_type = 0  # Regular move type
            
            # NOTE: We no longer need to rotate positions here because NardeEnv already
            # provides the board and valid moves from the current player's perspective.
            # The environment will handle the rotation when executing the move.
            
            action_code = from_pos * 24 + (0 if to_pos == 'off' else to_pos)
            valid_action_indices.append((action_code, move_type))
        
        # Filter probabilities to only valid actions
        valid_probs = np.array([action_probs[a[0]] for a in valid_action_indices])
        valid_probs /= valid_probs.sum()  # Normalize
        
        # Choose action based on probabilities
        chosen_idx = np.random.choice(len(valid_action_indices), p=valid_probs)
        return valid_action_indices[chosen_idx]
    
    def remember(self, state, action, reward):
        """Store episode history."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def update_policy(self):
        """Update policy network using REINFORCE algorithm."""
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.policy.device)
        actions = torch.tensor([a[0] for a in self.actions], device=self.policy.device)
        
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.policy.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Calculate loss
        action_probs = self.policy(states)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(selected_probs)
        loss = -(log_probs * discounted_rewards).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Clear episode history
        self.states = []
        self.actions = []
        self.rewards = []
        
        return loss.item()

# Reward shaping functions
def compute_block_reward(board):
    """
    Compute a bonus for consecutive points occupied by at least 2 checkers.
    A simple measure: count consecutive points where board[i] >= 2.
    Returns a bonus value (the higher the streak, the higher the bonus).
    """
    block_bonus = 0
    consecutive_count = 0
    for i in range(len(board)):
        if board[i] >= 2:
            consecutive_count += 1
        else:
            if consecutive_count > 1:
                block_bonus += (consecutive_count ** 1.5)
            consecutive_count = 0
    if consecutive_count > 1:
        block_bonus += (consecutive_count ** 1.5)
    return block_bonus

def compute_coverage_reward(board):
    """
    Reward spreading out your checkers.
    Here, we count the number of distinct points occupied by at least 2 checkers.
    """
    distinct_points = sum(1 for point in board if point >= 2)
    return distinct_points

def compute_progress(game, current_player):
    """
    Compute the remaining progress (total remaining pip distance) and the number of checkers borne off.
    In WHITE-ONLY perspective, the board is always from white's perspective.
    
    Returns:
        remaining_distance: Sum over board positions = sum_{i=0}^{23} (i * count at that point)
                           (Lower is better.)
        borne_off: Number of checkers borne off.
    """
    # In WHITE-ONLY perspective, the board is always from white's perspective
    # When current_player is -1, the board has already been rotated
    board = game.board
    
    # For the current player, checkers are always positive in WHITE-ONLY perspective
    remaining_distance = sum(i * max(0, board[i]) for i in range(24))
    
    # In WHITE-ONLY perspective, borne_off_white is always the current player's borne off count
    borne_off = game.borne_off_white
    
    return remaining_distance, borne_off

def train_candidate(candidate_index, episodes, learning_rate, save_dir, save_metrics=False,
                     max_steps=600, gamma=0.99, debug=False):
    """
    Trains a single candidate model using policy gradient (REINFORCE) and saves it to disk.
    Returns the path to the saved candidate model.
    """
    print(f"Pretraining candidate {candidate_index+1} ...")
    # Initialize a fresh agent
    agent = REINFORCEAgent(state_size=28, action_size=576, learning_rate=learning_rate, gamma=gamma)
    
    # Create an environment for pretraining
    env = gym.make('Narde-v0', render_mode=None, max_steps=5000, debug=debug)
    
    # Track metrics for verbosity
    episode_rewards = []
    losses = []
    
    # Maximum steps per episode
    max_steps_per_episode = max_steps
    
    # Training loop for this candidate:
    for ep in range(episodes):
        # Handle both old and new gym API
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result
        else:
            state = reset_result
        
        done = False
        total_reward = 0
        
        # Initialize progress metrics from the game
        prev_distance, prev_borne_off = compute_progress(env.unwrapped.game,
                                                         env.unwrapped.current_player)
        progress_weight = 0.01   # Increased reward per pip improvement (10x)
        borne_off_weight = 1.0   # Increased reward per additional checker borne off (10x)

        step_count = 0
        consecutive_skips = 0
        last_borne_off_step = 0
        
        while not done:
            step_count += 1
            # Sample dice for this turn
            dice = [np.random.randint(1, 7), np.random.randint(1, 7)]
            # Set the dice in the environment
            env.unwrapped.dice = dice
            
            if debug:
                print(f"\nStep {step_count} - Player: {env.unwrapped.current_player}")
                print(f"Dice: {dice}")
                board = env.unwrapped.game.board
                print("Board state:")
                for i in range(0, 24, 6):
                    print(board[i:i+6])
                print(f"White borne off: {env.unwrapped.game.borne_off_white}, Black borne off: {env.unwrapped.game.borne_off_black}")
            
            valid_moves = env.unwrapped.game.get_valid_moves(dice, env.unwrapped.current_player)
            
            # Check if any moves are bearing off moves
            bearing_off_moves = [move for move in valid_moves if move[1] == 'off']
            
            if debug:
                print(f"Valid moves: {valid_moves}")
                if bearing_off_moves:
                    print(f"Bearing off moves available: {bearing_off_moves}")
            
            if len(valid_moves) == 0:
                action = (0, 0)  # Skip turn if no valid moves
                consecutive_skips += 1
                if debug:
                    print(f"No valid moves, skipping turn. Consecutive skips: {consecutive_skips}")
            else:
                # If bearing off moves are available, increase their probability
                if bearing_off_moves and random.random() < 0.8:  # 80% chance to prioritize bearing off
                    # Choose a random bearing off move
                    move = random.choice(bearing_off_moves)
                    from_pos, to_pos = move
                    
                    # Convert to action format
                    if env.unwrapped.current_player == -1:
                        from_pos = 23 - from_pos
                    
                    action_code = from_pos * 24
                    action = (action_code, 1)  # 1 for bearing off
                    
                    if debug:
                        print(f"Prioritizing bearing off move: {move} -> action: {action}")
                else:
                    action = agent.act(state, valid_moves=valid_moves, env=env,
                                       dice=dice, current_player=env.unwrapped.current_player)
                
                consecutive_skips = 0
                if debug:
                    print(f"Chosen action: {action}")
            
            # Capture OLD board before stepping
            old_board = env.unwrapped.game.get_perspective_board(env.unwrapped.current_player)
            old_head_count = old_board[23]
            old_borne_off_white = env.unwrapped.game.borne_off_white
            old_borne_off_black = env.unwrapped.game.borne_off_black
            
            # Step the environment with the chosen action
            step_result = env.step(action)
            
            # Handle both old and new gym API
            if len(step_result) == 5:  # New API: (next_state, reward, done, truncated, info)
                next_state, env_reward, done, truncated, _ = step_result
                done = done or truncated
            else:  # Old API: (next_state, reward, done, info)
                next_state, env_reward, done, _ = step_result
            
            # Check if a checker was borne off
            new_borne_off_white = env.unwrapped.game.borne_off_white
            new_borne_off_black = env.unwrapped.game.borne_off_black
            
            if (new_borne_off_white > old_borne_off_white) or (new_borne_off_black > old_borne_off_black):
                last_borne_off_step = step_count
                if debug:
                    print(f"Checker borne off! White: {new_borne_off_white}, Black: {new_borne_off_black}")
            
            if debug:
                print(f"Reward: {env_reward}, Done: {done}")
                if consecutive_skips >= 3:
                    print("WARNING: 3 consecutive skips detected!")
            
            # Capture NEW board after stepping
            new_board = env.unwrapped.game.get_perspective_board(env.unwrapped.current_player)
            new_head_count = new_board[23]
            
            # Get new progress metrics
            new_distance, new_borne_off = compute_progress(env.unwrapped.game,
                                                           env.unwrapped.current_player)
            progress_reward = progress_weight * (prev_distance - new_distance)
            borne_off_increment = new_borne_off - prev_borne_off
            borne_reward = borne_off_weight * borne_off_increment
            current_board = env.unwrapped.game.get_perspective_board(env.unwrapped.current_player)
            
            # Additional shaping signals
            old_coverage = compute_coverage_reward(current_board)
            old_block = compute_block_reward(current_board)
            new_block = compute_block_reward(current_board)
            block_reward = (new_block - old_block)
            head_reward = 0.0
            if new_head_count < old_head_count:
                head_reward = 0.2 * (old_head_count - new_head_count)  # Increased reward (4x)
            coverage_weight = 0.001
            
            # Reward for moving checkers toward bearing off position (lower indices)
            home_board_reward = 0.0
            for i in range(6):  # First 6 positions (home board)
                home_board_reward += 0.02 * max(0, current_board[i])  # Reward for checkers in home board
            
            # Combine all rewards
            shaped_reward = env_reward + progress_reward + borne_reward \
                             + coverage_weight * old_coverage + 0.0001 * block_reward + head_reward \
                             + home_board_reward
            
            # Update progress metrics
            prev_distance, prev_borne_off = new_distance, new_borne_off
            
            # Store experience
            agent.remember(state, action, shaped_reward)
            total_reward += shaped_reward
            state = next_state
            
            # Debug information
            if (ep == 0 and step_count % 10 == 0) or debug:
                print(f"  Step {step_count}: Episode in progress")
            
            # Early termination if no progress in bearing off for a long time
            if step_count > 400 and step_count - last_borne_off_step > 300:
                if debug:
                    print(f"  Terminating episode due to lack of progress in bearing off")
                done = True
                    
            # Limit steps per episode to avoid infinite loops
            if step_count >= max_steps_per_episode:
                print(f"  Episode {ep+1} reached step limit of {max_steps_per_episode}")
                done = True
        
        # Update policy after episode
        loss = agent.update_policy()
        losses.append(loss)
        
        # Track metrics
        episode_rewards.append(total_reward)
        
        # Print progress every 10 episodes
        if (ep + 1) % 10 == 0 or ep == 0:
            avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            avg_loss = sum(losses[-10:]) / min(10, len(losses)) if losses else 0
            print(f"Candidate {candidate_index+1} - Episode {ep+1}/{episodes}, Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.6f}")
            print(f"  Steps in last episode: {step_count}, Learning rate: {agent.learning_rate:.6f}")
    
    # Print final training summary
    if episode_rewards:
        final_avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
        print(f"Candidate {candidate_index+1} - Final Avg Reward: {final_avg_reward:.4f}")
    
    # Calculate and print loss statistics if available
    if losses:
        initial_losses = losses[:min(5, len(losses))]
        final_losses = losses[-min(5, len(losses)):]
        
        initial_avg_loss = sum(initial_losses) / len(initial_losses)
        final_avg_loss = sum(final_losses) / len(final_losses)
        
        loss_reduction = initial_avg_loss - final_avg_loss
        loss_reduction_percent = (loss_reduction / initial_avg_loss * 100) if initial_avg_loss > 0 else 0
        
        print(f"Candidate {candidate_index+1} - Initial Avg Loss: {initial_avg_loss:.6f}")
        print(f"Candidate {candidate_index+1} - Final Avg Loss: {final_avg_loss:.6f}")
        print(f"Candidate {candidate_index+1} - Loss Reduction: {loss_reduction:.6f} ({loss_reduction_percent:.2f}%)")
    else:
        print(f"Candidate {candidate_index+1} - No loss data collected")
    
    # Save metrics if requested
    if save_metrics:
        metrics_dir = os.path.join(save_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, f"candidate_{candidate_index+1}_metrics.npz")
        np.savez(
            metrics_path,
            episode_rewards=np.array(episode_rewards),
            losses=np.array(losses)
        )
        print(f"Candidate {candidate_index+1} metrics saved to {metrics_path}")
    
    env.close()
    candidate_path = os.path.join(save_dir, f"candidate_{candidate_index+1}.pt")
    torch.save(agent.policy.state_dict(), candidate_path)
    print(f"Candidate {candidate_index+1} saved to {candidate_path}")
    return candidate_path

def pretrain_candidates(candidate_count, episodes, learning_rate, save_dir, parallel_candidates, save_metrics=False,
                     max_steps=600, gamma=0.99, debug=False):
    """
    Pretrain multiple candidate models using policy gradient (REINFORCE).
    
    Args:
        candidate_count: Number of candidate models to generate
        episodes: Number of training episodes per candidate
        learning_rate: Learning rate for gradient descent
        save_dir: Directory to save candidate models
        parallel_candidates: Number of candidates to train in parallel
        save_metrics: Whether to save training metrics
        max_steps: Maximum steps per episode
        gamma: Discount factor
        debug: Whether to print debug information
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Create argument list for each candidate.
    args_list = [(i, episodes, learning_rate, save_dir, save_metrics, max_steps, gamma, debug) 
                for i in range(candidate_count)]
    
    if parallel_candidates > 1 and not debug:
        pool = multiprocessing.Pool(processes=parallel_candidates)
        candidate_paths = pool.starmap(train_candidate, args_list)
        pool.close()
        pool.join()
    else:
        candidate_paths = [train_candidate(*args) for args in args_list]
    
    # Print summary
    print("\n" + "="*50)
    print(f"PRETRAINING SUMMARY")
    print("="*50)
    print(f"Total candidates: {len(candidate_paths)}")
    print(f"Episodes per candidate: {episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Gamma: {gamma}")
    print(f"Debug mode: {debug}")
    print(f"Models saved to: {save_dir}")
    print("="*50)
    
    print(f"Pretraining complete: {len(candidate_paths)} candidate models saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain candidate models using policy gradient (REINFORCE) and save them."
    )
    parser.add_argument('--candidate-count', type=int, default=50, help="Number of candidate models to generate")
    parser.add_argument('--episodes', type=int, default=1000, help="Number of training episodes per candidate")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for gradient descent")
    parser.add_argument('--save-dir', type=str, default="pretrained_models", help="Folder to save candidate models")
    parser.add_argument('--parallel-candidates', type=int, default=1,
                        help="Number of candidate models to pretrain in parallel")
    parser.add_argument('--save-metrics', action='store_true', help="Save training metrics to files")
    parser.add_argument('--max-steps', type=int, default=600, help="Maximum steps per episode")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode with verbose output")
    
    args = parser.parse_args()
    
    pretrain_candidates(
        candidate_count=args.candidate_count,
        episodes=args.episodes,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        parallel_candidates=args.parallel_candidates,
        save_metrics=args.save_metrics,
        max_steps=args.max_steps,
        gamma=args.gamma,
        debug=args.debug
    )
