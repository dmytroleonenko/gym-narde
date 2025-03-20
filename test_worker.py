import os
import torch
from muzero.parallel_training import optimized_self_play_worker

def test_worker():
    # Just check if we can import the function
    print("Successfully imported worker function")
    print("Function location:", optimized_self_play_worker.__module__)
    
    # Create dummy weights
    dummy_network = torch.nn.Linear(10, 10)
    dummy_weights = dummy_network.state_dict()
    
    # Call the function with minimal args just to test imports
    try:
        # We expect this to fail at some point but we just want to
        # test that the function can be imported and is callable
        optimized_self_play_worker(
            worker_id=0,
            network_weights=dummy_weights,
            num_games=1,
            save_dir="/tmp",
            num_simulations=1,
            temperature=1.0,
            temperature_drop=None,
            mcts_batch_size=1,
            env_batch_size=1,
            hidden_dim=64,
            target_device="cpu"
        )
    except Exception as e:
        print(f"Expected error when running (but function was callable): {e}")
    
    print("Test complete - worker function can be imported and called")

if __name__ == "__main__":
    test_worker() 