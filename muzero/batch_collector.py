"""
Batch collector for efficient MuZero training with MPS acceleration.
Pre-generates game steps and stores them for processing in large batches.
"""

import torch
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
import threading
import queue
import time
import gym_narde  # Import to register the environment
from muzero.mcts import MCTS
from muzero.models import MuZeroNetwork

# Define Step type for storing generated steps
Step = namedtuple('Step', ['observation', 'action', 'reward', 'next_observation', 'done', 'info'])

class BatchStepCollector:
    """Collects steps from environments to be processed in batches on MPS."""
    
    def __init__(self, batch_size=2048, num_parallel_envs=16, device="cpu"):
        self.batch_size = batch_size
        self.num_parallel_envs = num_parallel_envs
        self.collection_device = torch.device("cpu")  # Always collect on CPU
        self.training_device = torch.device(device)   # Process on target device
        
        # Create environment to get dimensions
        env = gym.make('Narde-v0')
        self.input_dim = env.observation_space.shape[0]
        self.action_dim = 24 * 24  # 576 possible (from, to) combinations
        env.close()
        
        # Create model for self-play (on CPU for data generation)
        self.model = MuZeroNetwork(self.input_dim, self.action_dim).to(self.collection_device)
        
        # Create buffer for collected steps
        self.step_buffer = []
        
        # Create parallel environments
        self.envs = [gym.make('Narde-v0') for _ in range(num_parallel_envs)]
        
        # Create MCTSs for each environment
        self.mcts_list = [
            MCTS(
                network=self.model,
                num_simulations=50,
                discount=0.997,
                action_space_size=self.action_dim,
                device=self.collection_device
            ) for _ in range(num_parallel_envs)
        ]
        
        # Initialize environment states
        self.observations = []
        for env in self.envs:
            observation, _ = env.reset()
            self.observations.append(observation)
    
    def collect_steps(self, num_steps):
        """Collect a specific number of steps from environments."""
        steps_collected = 0
        
        while steps_collected < num_steps:
            # Process each environment
            for i in range(self.num_parallel_envs):
                if steps_collected >= num_steps:
                    break
                
                env = self.envs[i]
                observation = self.observations[i]
                mcts = self.mcts_list[i]
                
                # Get valid actions for this environment
                unwrapped_env = env.unwrapped
                valid_moves = unwrapped_env.game.get_valid_moves()
                valid_actions = []
                
                for move in valid_moves:
                    # Convert move to action index
                    if move[1] == 'off':
                        # Bear off move: (from_pos, 'off')
                        action_idx = move[0] * 24
                    else:
                        # Regular move: (from_pos, to_pos)
                        action_idx = move[0] * 24 + move[1]
                    
                    valid_actions.append(action_idx)
                
                # Run MCTS to select action
                if len(valid_actions) > 0:
                    mcts_probs = mcts.run(observation, valid_actions, add_exploration_noise=True)
                    action = valid_actions[np.argmax(mcts_probs)]
                    
                    # Convert action index back to move
                    from_pos = action // 24
                    to_pos = action % 24
                    
                    if to_pos == 0:
                        # Bear off move
                        move = (from_pos, 'off')
                    else:
                        # Regular move
                        move = (from_pos, to_pos)
                    
                    # Take step in environment
                    next_observation, reward, done, truncated, info = env.step(move)
                    
                    # Store step
                    self.step_buffer.append(Step(
                        observation=observation,
                        action=action,
                        reward=reward,
                        next_observation=next_observation,
                        done=done or truncated,
                        info=info
                    ))
                    
                    # Update observation
                    self.observations[i] = next_observation if not (done or truncated) else env.reset()[0]
                    
                    steps_collected += 1
                else:
                    # No valid actions, reset environment
                    self.observations[i] = env.reset()[0]
        
        print(f"Collected {steps_collected} steps")
        return steps_collected
    
    def get_batch(self, batch_size=None):
        """Get a batch of steps for processing."""
        if batch_size is None:
            batch_size = self.batch_size
        
        # If we don't have enough steps, collect more
        if len(self.step_buffer) < batch_size:
            self.collect_steps(batch_size - len(self.step_buffer))
        
        # Get batch
        batch = self.step_buffer[:batch_size]
        self.step_buffer = self.step_buffer[batch_size:]
        
        # Convert to tensors
        observations = torch.FloatTensor([step.observation for step in batch]).to(self.training_device).to(torch.bfloat16)
        actions = torch.LongTensor([step.action for step in batch]).to(self.training_device)
        rewards = torch.FloatTensor([step.reward for step in batch]).to(self.training_device).to(torch.bfloat16)
        next_observations = torch.FloatTensor([step.next_observation for step in batch]).to(self.training_device).to(torch.bfloat16)
        dones = torch.BoolTensor([step.done for step in batch]).to(self.training_device)
        
        return observations, actions, rewards, next_observations, dones
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


class AsyncBatchCollector:
    """
    Asynchronously collects experience in the background while the model trains.
    This allows for better utilization of CPU (for collection) and GPU/MPS (for training).
    """
    
    def __init__(self, batch_size=2048, buffer_size=8192, num_parallel_envs=16, device="mps"):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.collector = BatchStepCollector(batch_size, num_parallel_envs, device)
        self.buffer = queue.Queue(maxsize=buffer_size)  # Thread-safe queue
        self.collection_thread = None
        self.stop_collection = False
    
    def _collection_worker(self):
        """Background worker that continuously collects experience."""
        while not self.stop_collection:
            # Check if buffer has space
            if self.buffer.qsize() < self.buffer_size - self.batch_size:
                # Collect steps
                self.collector.collect_steps(self.batch_size)
                
                # Get batch and add to buffer
                batch = self.collector.get_batch(self.batch_size)
                try:
                    self.buffer.put(batch, block=False)
                except queue.Full:
                    # If buffer is full, wait a bit
                    time.sleep(0.1)
            else:
                # Buffer is nearly full, wait
                time.sleep(0.1)
    
    def start_collection(self):
        """Start asynchronous collection."""
        if self.collection_thread is None or not self.collection_thread.is_alive():
            self.stop_collection = False
            self.collection_thread = threading.Thread(target=self._collection_worker)
            self.collection_thread.daemon = True  # Thread will exit when main program exits
            self.collection_thread.start()
            print("Started asynchronous experience collection")
    
    def stop_collection(self):
        """Stop asynchronous collection."""
        self.stop_collection = True
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)
        print("Stopped asynchronous experience collection")
    
    def get_batch(self, block=True, timeout=None):
        """Get a batch of experience from the buffer."""
        try:
            return self.buffer.get(block=block, timeout=timeout)
        except queue.Empty:
            # If no batch is available and non-blocking, return None
            return None
    
    def close(self):
        """Clean up resources."""
        self.stop_collection = True
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)
        self.collector.close() 