import os
import argparse
import torch
import gymnasium as gym
import numpy as np
import copy
from train_deepq_pytorch import DQNAgent, DecomposedDQN

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

def pretrain_candidates(candidate_count, episodes, learning_rate, save_dir):
    # Create save directory if needed.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # For pretraining, we create a base DQNAgent (or just the model) and train it for the specified number of episodes.
    # For simplicity, we’ll use the DQNAgent, run its training loop for a few episodes, and then save a copy.
    # (You can tweak the internal loop as needed.)
    
    candidates = []
    for i in range(candidate_count):
        print(f"Pretraining candidate {i+1}/{candidate_count} ...")
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
            # Using the current player's perspective (agent.env.unwrapped.current_player)
            prev_distance, prev_borne_off = compute_progress(agent.env.unwrapped.game,
                                                             agent.env.unwrapped.current_player)
            # Set hyperparameters for shaping rewards
            progress_weight = 0.001   # Reward per pip improvement
            borne_off_weight = 0.1    # Reward per additional checker borne off

            # Sample dice for this turn
            dice = [np.random.randint(1, 7), np.random.randint(1, 7)]
            # Get valid moves from the game logic
            valid_moves = agent.env.unwrapped.game.get_valid_moves(dice, agent.env.unwrapped.current_player)
                if len(valid_moves) == 0:
                    action = (0, 0)  # Skip turn if no valid moves
                else:
                    # Use agent.act to select an action with exploration (training=True)
                    action = agent.act(state, valid_moves=valid_moves, env=agent.env,
                                       dice=dice, current_player=agent.env.unwrapped.current_player,
                                       training=True)
                # Step the environment with the chosen action
                next_state, env_reward, done, truncated, _ = agent.env.step(action)
                # Get new progress metrics
                new_distance, new_borne_off = compute_progress(agent.env.unwrapped.game,
                                                               agent.env.unwrapped.current_player)
                # Compute incremental progress:
                # A reduction in remaining distance (i.e. progress) yields positive reward.
                progress_reward = progress_weight * (prev_distance - new_distance)
                # Reward any additional checkers borne off since previous step.
                borne_off_increment = new_borne_off - prev_borne_off
                borne_reward = borne_off_weight * borne_off_increment

                # Compute the final shaped reward by blending the environment reward and the incremental rewards.
                shaped_reward = env_reward + progress_reward + borne_reward

                # Update progress metrics for the next step
                prev_distance, prev_borne_off = new_distance, new_borne_off
                # Store the experience with shaped reward
                agent.remember(state, action, shaped_reward, next_state, done)
                total_reward += shaped_reward
                state = next_state
                # Train the agent if enough samples have been collected
                if len(agent.memory) >= agent.batch_size:
                    agent.replay()
            print(f"Candidate {i+1}, Episode {ep+1}/{episodes}: Total Reward = {total_reward:.2f}")
        
        # Optionally, close the environment when done training this candidate.
        agent.env.close()
            
        # After training, save the candidate model.
        candidate_path = os.path.join(save_dir, f"candidate_{i+1}.pt")
        torch.save(agent.model.state_dict(), candidate_path)
        print(f"Candidate {i+1} saved to {candidate_path}")
        candidates.append(candidate_path)
    
    print(f"Pretraining complete: {len(candidates)} candidate models saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain candidate models using DQN gradient descent and save them."
    )
    parser.add_argument('--candidate-count', type=int, default=50, help="Number of candidate models to generate")
    parser.add_argument('--episodes', type=int, default=1000, help="Number of training episodes per candidate")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for gradient descent")
    parser.add_argument('--save-dir', type=str, default="pretrained_models", help="Folder to save candidate models")
    
    args = parser.parse_args()
    
    pretrain_candidates(
        candidate_count=args.candidate_count,
        episodes=args.episodes,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
