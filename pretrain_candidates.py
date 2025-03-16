import os
import argparse
import torch
import gymnasium as gym
import numpy as np
import copy
import multiprocessing
from train_deepq_pytorch import DQNAgent, DecomposedDQN

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

def compute_head_reward(board, head_index):
    # Disable the old penalty entirely:
    return 0

def compute_progress(game, current_player):
    """
    Compute the remaining progress (total remaining pip distance) and the number of checkers borne off.
    We assume the state is presented in the current player's perspective (i.e. lower indices mean closer to bearing off).
    
    Returns:
        remaining_distance: Sum over board positions = sum_{i=0}^{23} (i * count at that point)
                           (Lower is better.)
        borne_off: Number of checkers borne off.
    """
    # Obtain the board from the game: in current player's perspective.
    board = game.get_perspective_board(current_player)
    remaining_distance = sum(i * max(0, board[i]) for i in range(24))
    # For both players, when using perspective, checkers are positive.
    # Assume borne_off is stored in the game as follows:
    borne_off = game.borne_off_white if current_player == 1 else game.borne_off_black
    return remaining_distance, borne_off

def train_candidate(candidate_index, episodes, learning_rate, save_dir):
    """
    Trains a single candidate model and saves it to disk.
    Returns the path to the saved candidate model.
    """
    print(f"Pretraining candidate {candidate_index+1} ...")
    # Initialize a fresh agent with the decomposed network.
    agent = DQNAgent(
        state_size=28,
        action_size=576 * 576,
        use_decomposed_network=True,
        use_prioritized_replay=True
    )
    # Override learning rate if provided.
    agent.learning_rate = learning_rate
    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=learning_rate)
    
    # Create an environment for pretraining and attach it to the agent.
    agent.env = gym.make('Narde-v0', render_mode=None)
    
    # Training loop for this candidate:
    for ep in range(episodes):
        state, _ = agent.env.reset()
        done = False
        total_reward = 0
        # Initialize progress metrics from the game
        prev_distance, prev_borne_off = compute_progress(agent.env.unwrapped.game,
                                                         agent.env.unwrapped.current_player)
        progress_weight = 0.001   # Reward per pip improvement
        borne_off_weight = 0.1    # Reward per additional checker borne off

        while not done:
            # Sample dice for this turn
            dice = [np.random.randint(1, 7), np.random.randint(1, 7)]
            valid_moves = agent.env.unwrapped.game.get_valid_moves(dice, agent.env.unwrapped.current_player)
            if len(valid_moves) == 0:
                action = (0, 0)  # Skip turn if no valid moves
            else:
                action = agent.act(state, valid_moves=valid_moves, env=agent.env,
                                   dice=dice, current_player=agent.env.unwrapped.current_player,
                                   training=True)
            # Capture OLD board before stepping
            old_board = agent.env.unwrapped.game.get_perspective_board(agent.env.unwrapped.current_player)
            old_head_count = old_board[23]
            # Step the environment with the chosen action
            next_state, env_reward, done, truncated, _ = agent.env.step(action)
            done = done or truncated
            # Capture NEW board after stepping
            new_board = agent.env.unwrapped.game.get_perspective_board(agent.env.unwrapped.current_player)
            new_head_count = new_board[23]
            # Get new progress metrics
            new_distance, new_borne_off = compute_progress(agent.env.unwrapped.game,
                                                           agent.env.unwrapped.current_player)
            progress_reward = progress_weight * (prev_distance - new_distance)
            borne_off_increment = new_borne_off - prev_borne_off
            borne_reward = borne_off_weight * borne_off_increment
            current_board = agent.env.unwrapped.game.get_perspective_board(agent.env.unwrapped.current_player)
            # Additional shaping signals
            old_coverage = compute_coverage_reward(current_board)
            old_block = compute_block_reward(current_board)
            new_block = compute_block_reward(current_board)
            block_reward = (new_block - old_block)
            head_reward = 0.0
            if new_head_count < old_head_count:
                head_reward = 0.05 * (old_head_count - new_head_count)
            coverage_weight = 0.0005
            shaped_reward = env_reward + progress_reward + borne_reward \
                             + coverage_weight * old_coverage + 0.0001 * block_reward + head_reward
            prev_distance, prev_borne_off = new_distance, new_borne_off
            agent.remember(state, action, shaped_reward, next_state, done)
            total_reward += shaped_reward
            state = next_state
            if len(agent.memory) >= agent.batch_size:
                agent.replay()
    agent.env.close()
    candidate_path = os.path.join(save_dir, f"candidate_{candidate_index+1}.pt")
    torch.save(agent.model.state_dict(), candidate_path)
    print(f"Candidate {candidate_index+1} saved to {candidate_path}")
    return candidate_path

def pretrain_candidates(candidate_count, episodes, learning_rate, save_dir, parallel_candidates):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Create argument list for each candidate.
    args_list = [(i, episodes, learning_rate, save_dir) for i in range(candidate_count)]
    
    if parallel_candidates > 1:
        pool = multiprocessing.Pool(processes=parallel_candidates)
        candidate_paths = pool.map(lambda args: train_candidate(*args), args_list)
        pool.close()
        pool.join()
    else:
        candidate_paths = [train_candidate(i, episodes, learning_rate, save_dir) for i in range(candidate_count)]
    
    print(f"Pretraining complete: {len(candidate_paths)} candidate models saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain candidate models using DQN gradient descent and save them."
    )
    parser.add_argument('--candidate-count', type=int, default=50, help="Number of candidate models to generate")
    parser.add_argument('--episodes', type=int, default=1000, help="Number of training episodes per candidate")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for gradient descent")
    parser.add_argument('--save-dir', type=str, default="pretrained_models", help="Folder to save candidate models")
    
    parser.add_argument('--parallel-candidates', type=int, default=1,
                        help="Number of candidate models to pretrain in parallel")
    
    args = parser.parse_args()
    
    pretrain_candidates(
        candidate_count=args.candidate_count,
        episodes=args.episodes,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        parallel_candidates=args.parallel_candidates
    )
    
    pretrain_candidates(
        candidate_count=args.candidate_count,
        episodes=args.episodes,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
