import numpy as np
import jax
import jax.numpy as jnp
from timeit import default_timer as timer
from optimized_narde_env import OptimizedNardeEnv, simulate_moves_batch_numpy

class MuZeroNardeInterface:
    """
    Interface for MuZero to interact with the optimized Narde environment.
    This class makes decisions about when to use JAX vs. NumPy based on batch sizes.
    """
    
    def __init__(self, rotation_threshold=4096, block_rule_threshold=128, nn_threshold=512):
        """
        Initialize the MuZero interface with optimal thresholds for different operations.
        
        Args:
            rotation_threshold: Batch size threshold for switching from NumPy to JAX for board rotation
            block_rule_threshold: Batch size threshold for switching from NumPy to JAX for block rule checking
            nn_threshold: Batch size threshold for switching from JAX to PyTorch for neural network operations
        """
        self.env = OptimizedNardeEnv()
        self.rotation_threshold = rotation_threshold  # For board rotation operations
        self.block_rule_threshold = block_rule_threshold  # For block rule checking
        self.nn_threshold = nn_threshold  # For neural network operations
        
        print(f"MuZero Narde Interface initialized with thresholds:")
        print(f"  • Board rotation: JAX for batch sizes ≥ {rotation_threshold}")
        print(f"  • Block rule: JAX for batch sizes ≥ {block_rule_threshold}")
        print(f"  • Neural networks: JAX for batch sizes < {nn_threshold}, PyTorch for larger batches")
        
    def get_initial_state(self):
        """Get the initial state of the game."""
        # reset() returns (observation, info) in Gym environments
        observation, _ = self.env.reset()
        return observation
    
    def get_valid_actions(self, state, dice=None):
        """Get valid actions for a state."""
        self.env.board = np.array(state)
        if dice is not None:
            self.env.dice = dice
        # Call internal method to get valid actions
        self.env.valid_actions = self.env._get_valid_actions()
        return self.env.valid_actions
    
    def step(self, state, action):
        """Take a step in the environment."""
        self.env.board = np.array(state)
        # step() returns (observation, reward, terminated, truncated, info) in newer Gym environments
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def get_canonical_state(self, state, player):
        """Get the canonical state for a player."""
        # In Narde, rotate the board 180 degrees for player 2
        if player == 2:  # For player 2, rotate the board
            return self.env._rotate_board(np.array(state))
        return np.array(state)
    
    def simulate_batch(self, states, actions, batch_size):
        """
        Simulate a batch of moves, choosing between JAX and NumPy based on optimal batch size threshold.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            batch_size: Size of the batch
            
        Returns:
            next_states: Batch of next states
            rewards: Batch of rewards
            dones: Batch of done flags
        """
        start_time = timer()
        
        # Convert to numpy arrays to ensure consistency
        states_np = np.array(states)
        actions_np = np.array(actions)
        
        # Choose between JAX and NumPy based on batch size - using our benchmark findings
        # For batch simulation, JAX implementation is still being developed, so we use NumPy for now
        next_states, rewards, dones = simulate_moves_batch_numpy(states_np, actions_np)
        method = "NumPy"
        
        end_time = timer()
        execution_time = end_time - start_time
        
        if batch_size > 1:  # Only print for actual batches
            print(f"Batch simulation (size {batch_size}) using {method}: {execution_time:.6f}s")
        
        return next_states, rewards, dones
    
    def check_block_rule_batch(self, states, batch_size):
        """
        Check block rule for a batch of states, choosing between JAX and NumPy based on optimal threshold.
        
        Args:
            states: Batch of states
            batch_size: Size of the batch
            
        Returns:
            Array of booleans indicating whether block rule is violated
        """
        start_time = timer()
        
        # Convert to numpy arrays to ensure consistency
        states_np = np.array(states)
        
        # Choose between JAX and NumPy based on batch size
        if batch_size >= self.block_rule_threshold:
            # Use JAX for large batches
            # The OptimizedNardeEnv has a check_block_rule_batch method for batched operations
            results = self.env.check_block_rule_batch(states_np)
            method = "JAX"
        else:
            # Use NumPy for small batches
            results = np.zeros(batch_size, dtype=bool)
            for i in range(batch_size):
                results[i] = self.env.check_block_rule(states_np[i])
            method = "NumPy"
        
        end_time = timer()
        execution_time = end_time - start_time
        
        if batch_size > 1:  # Only print for actual batches
            print(f"Block rule check (size {batch_size}) using {method}: {execution_time:.6f}s")
        
        return results
    
    def rotate_boards_batch(self, states, batch_size):
        """
        Rotate a batch of boards, choosing between JAX and NumPy based on optimal threshold.
        
        Args:
            states: Batch of states
            batch_size: Size of the batch
            
        Returns:
            Rotated boards
        """
        start_time = timer()
        
        # Convert to numpy arrays to ensure consistency
        states_np = np.array(states)
        
        # Choose between JAX and NumPy based on batch size
        if batch_size >= self.rotation_threshold:
            # Use JAX for very large batches (4096+)
            states_jax = jnp.array(states_np)
            rotated_boards = -jnp.flip(states_jax, axis=1)
            # Convert back to numpy
            result = np.array(rotated_boards)
            method = "JAX"
        else:
            # Use NumPy for small and medium batches (faster until ~4096)
            result = -np.flip(states_np, axis=1)
            method = "NumPy"
        
        end_time = timer()
        execution_time = end_time - start_time
        
        if batch_size > 1:  # Only print for actual batches
            print(f"Board rotation (size {batch_size}) using {method}: {execution_time:.6f}s")
        
        return result
    
    def render_state(self, state):
        """Render a state."""
        self.env.board = np.array(state)
        return self.env.render()


def test_muzero_interface():
    """Test the MuZero interface."""
    print("=== Testing MuZero Interface ===")
    
    # Initialize the interface
    interface = MuZeroNardeInterface(rotation_threshold=4096, block_rule_threshold=128, nn_threshold=512)
    
    # Get initial state
    state = interface.get_initial_state()
    print(f"Initial state shape: {state.shape}")
    
    # Get valid actions
    valid_actions = interface.get_valid_actions(state)
    print(f"Valid actions: {valid_actions[:5]}... (total: {len(valid_actions)})")
    
    # Test step
    next_state, reward, done, info = interface.step(state, valid_actions[0])
    print(f"After step: reward={reward}, done={done}")
    
    # Test canonical state
    canonical_state = interface.get_canonical_state(state, 2)
    print(f"Canonical state for player 2 shape: {canonical_state.shape}")
    
    # Test batch simulation with different batch sizes
    small_batch_sizes = [1, 8, 32, 128, 512]
    large_batch_sizes = [1024, 2048]
    extra_large_batch_sizes = [4096, 8192, 16384]
    batch_sizes = small_batch_sizes + large_batch_sizes + extra_large_batch_sizes
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Create test data
        states = np.tile(state, (batch_size, 1))
        actions = np.array([valid_actions[0]] * batch_size)
        
        # Test board rotation batch operation
        if batch_size > 1:  # Skip for single state
            rotated_states = interface.rotate_boards_batch(states, batch_size)
            print(f"Board rotation completed with shape: {rotated_states.shape}")
        
        # Test simulation
        next_states, rewards, dones = interface.simulate_batch(states, actions, batch_size)
        print(f"Simulation results: {len(next_states)} next states, {sum(dones)} games finished")
        
        # Test block rule check
        block_violated = interface.check_block_rule_batch(states, batch_size)
        print(f"Block rule violated: {sum(block_violated)}/{batch_size}")
    
    print("\n=== MuZero Interface Test Complete ===")


if __name__ == "__main__":
    test_muzero_interface() 