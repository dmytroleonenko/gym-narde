#!/usr/bin/env python3
"""
Train a DQN agent for Narde game using the DecomposedDQN architecture.
"""

import os
import argparse
import multiprocessing as mp
import torch
import matplotlib.pyplot as plt
import gymnasium as gym

# Force MPS device if available
if torch.backends.mps.is_available():
    # Only print in the main process
    if mp.current_process().name == 'MainProcess':
        print("MPS device found - using basic MPS configuration")
    
    # Enable compatibility mode for MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Remove any custom MPS memory settings that might be causing issues
    if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
        del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
else:
    if mp.current_process().name == 'MainProcess':
        print("MPS device not available, using CPU")

# Import from our package
from narde_rl.networks.decomposed_dqn import DecomposedDQN
from narde_rl.utils.agents import DQNAgent
from narde_rl.utils.training import (
    test_action_conversion, 
    ParallelEnv, 
    train
)

def main():
    """
    Main function to train a DecomposedDQN-based DQN agent for Narde game.
    """
    print("Starting Narde training with DecomposedDQN architecture...")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a DecomposedDQN agent for Narde')
    parser.add_argument('--episodes', type=int, default=1000, help="Number of episodes to train")
    parser.add_argument('--max-steps', type=int, default=500, help="Maximum steps per episode")
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode")
    parser.add_argument('--verbose', action='store_true', help="Print verbose debug information")
    parser.add_argument('--num-envs', type=int, default=2, help="Number of parallel environments")
    parser.add_argument('--test-actions', action='store_true', help="Run action conversion test only")
    
    args = parser.parse_args()
    print(f"Configuration: episodes={args.episodes}, max_steps={args.max_steps}, epsilon_decay={args.epsilon_decay}, num_envs={args.num_envs}")
    
    # Run action conversion test if requested
    if args.test_actions:
        print("Running action conversion test...")
        test_action_conversion()
        return
    
    # Initialize with default values
    lr = 1e-4  # Lower learning rate for stability
    epsilon_start = 1.0
    epsilon_min = 0.01
    
    # Create directory for saved models
    os.makedirs('saved_models', exist_ok=True)
    
    # Environment setup
    print("Initializing environment...")
    state_dim = 28  # Observation space size
    action_dim = 576  # Max possible moves
    parallel_env = ParallelEnv(env_name='Narde-v0', num_envs=args.num_envs)
    
    # Create DecomposedDQN agent
    print("Creating DecomposedDQN agent...")
    agent = DQNAgent(
        state_size=state_dim,
        action_size=action_dim,
        network_class=DecomposedDQN,
        use_decomposed=True
    )
    
    # Update agent parameters for better MPS compatibility
    agent.epsilon = epsilon_start
    agent.epsilon_min = epsilon_min
    agent.epsilon_decay = args.epsilon_decay
    agent.learning_rate = lr
    
    # For MPS, use very conservative settings
    if agent.device.type == "mps":
        agent.batch_size = 128  # Smaller batch size for MPS
        agent.grad_accumulation_steps = 1  # Disable gradient accumulation for MPS
        print(f"Using MPS-optimized settings: batch_size={agent.batch_size}, no gradient accumulation")
    else:
        agent.batch_size = 512  # Larger batch size for CUDA/CPU
        
    agent.update_target_freq = 5  # More frequent target network updates
    
    print(f"Agent created: epsilon={agent.epsilon}, learning_rate={agent.learning_rate}")
    print(f"Using device: {agent.device} with batch size: {agent.batch_size} and optimized MPS settings")

    try:
        print("Starting training...")
        # Train agent using our train function
        episode_rewards = train(
            agent, 
            parallel_env, 
            episodes=args.episodes, 
            max_steps=args.max_steps,
            epsilon_decay=args.epsilon_decay,
            debug=args.debug,
            verbose=args.verbose
        )
        
        # Save final model
        print("Training complete, saving model...")
        model_path = 'saved_models/decomposeddqn_narde_final.pt'
        torch.save(agent.model.network.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Plot training progress
        print("Generating training progress plot...")
        plot_path = 'saved_models/decomposeddqn_training_progress.png'
        plt.figure(figsize=(12, 4))
        plt.plot(range(len(episode_rewards)), episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('DecomposedDQN Episode Rewards during Training')
        plt.savefig(plot_path)
        print(f"Training progress plot saved to {plot_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
        interrupted_path = 'saved_models/decomposeddqn_narde_interrupted.pt'
        torch.save(agent.model.network.state_dict(), interrupted_path)
        print(f"Model saved to {interrupted_path}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Make sure we always close the environment
        parallel_env.close()
        
    print("Training complete!")

if __name__ == "__main__":
    # Set number of threads for parallel processing
    mp.set_start_method('spawn')
    
    # Regular training
    main() 