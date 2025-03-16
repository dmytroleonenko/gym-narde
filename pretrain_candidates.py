import os
import argparse
import torch
import gymnasium as gym
import numpy as np
import copy
from train_deepq_pytorch import DQNAgent, DecomposedDQN

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
            while not done:
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
                next_state, reward, done, truncated, _ = agent.env.step(action)
                done = done or truncated
                # Store the experience
                agent.remember(state, action, reward, next_state, done)
                total_reward += reward
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
