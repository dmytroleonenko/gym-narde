"""
Batched Monte Carlo Tree Search implementation for MuZero.
This optimized version processes multiple simulations in parallel to reduce CPU-GPU synchronization.
"""

import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set, Any


class BatchedNode:
    """
    Node class for batched Monte Carlo Tree Search.
    Optimized to minimize CPU-GPU synchronization.
    """
    def __init__(self, prior: float = 0.0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children: Dict[int, 'BatchedNode'] = {}
        self.hidden_state = None
        self.reward = 0.0
        
    def expanded(self) -> bool:
        """Check if the node has been expanded."""
        return len(self.children) > 0
    
    def value(self) -> float:
        """Get the average value of the node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class BatchedMCTS:
    """
    Batched Monte Carlo Tree Search for MuZero.
    Optimized to minimize CPU-GPU transfers and maximize GPU usage.
    """
    def __init__(
        self,
        network,
        num_simulations: int,
        batch_size: int = 16,
        discount: float = 0.99,
        dirichlet_alpha: float = 0.3,
        exploration_fraction: float = 0.25,
        pb_c_base: int = 19652,
        pb_c_init: float = 1.25,
        action_space_size: int = 576,
        device: str = "cpu"
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.batch_size = batch_size  # How many simulations to batch together
        self.discount = discount
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.action_space_size = action_space_size
        self.device = device
        
        # Pre-allocate tensors for batched operations
        self.action_batch = None
        self.hidden_batch = None
        self._init_batched_tensors()
        
        # Cache valid action masks
        self._action_mask_cache = {}
        
    def _init_batched_tensors(self):
        """Initialize tensors for batched operations."""
        self.action_batch = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        # The hidden state size depends on the network's architecture
        hidden_dim = self.network.hidden_dim
        self.hidden_batch = torch.zeros(self.batch_size, hidden_dim, device=self.device)
        
    def run(self, observation, valid_actions=None, add_exploration_noise=True):
        """
        Run the MCTS algorithm on the given observation.
        
        Args:
            observation: The current observation
            valid_actions: List of valid actions (indices)
            add_exploration_noise: Whether to add Dirichlet noise at the root
            
        Returns:
            The search policy (visit counts for each action)
        """
        # Convert numpy observations to torch tensors
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation).to(self.device)
        
        # Single initial inference for root
        with torch.no_grad():
            root_hidden_state, root_value, root_policy_logits = self.network.initial_inference(observation.unsqueeze(0))
        
        # Create valid action mask on device
        if valid_actions is not None:
            # Use cached action mask if exists for these valid actions
            valid_actions_key = tuple(sorted(valid_actions))
            if valid_actions_key in self._action_mask_cache:
                action_mask = self._action_mask_cache[valid_actions_key]
            else:
                action_mask = torch.zeros(self.action_space_size, device=self.device)
                action_mask[valid_actions] = 1.0
                # Cache the mask for future use
                self._action_mask_cache[valid_actions_key] = action_mask
                
            # Apply mask to policy logits (set invalid actions to -inf)
            # Keep everything on GPU to avoid synchronization
            masked_policy_logits = root_policy_logits.clone()
            masked_policy_logits[0, ~action_mask.bool()] = float('-inf')
            root_policy = torch.softmax(masked_policy_logits, dim=1).squeeze(0)
        else:
            # If no valid actions provided, use all actions
            root_policy = torch.softmax(root_policy_logits, dim=1).squeeze(0)
        
        # Add exploration noise to the root node policy
        if add_exploration_noise:
            # Generate and apply noise directly on GPU
            if valid_actions is not None:
                noise = torch.zeros_like(root_policy)
                # Generate Dirichlet noise - requires CPU operations
                valid_noise = torch.tensor(
                    np.random.dirichlet([self.dirichlet_alpha] * len(valid_actions)),
                    dtype=torch.float32,
                    device=self.device
                )
                # Apply noise to valid actions only
                for i, action in enumerate(valid_actions):
                    noise[action] = valid_noise[i]
            else:
                noise = torch.tensor(
                    np.random.dirichlet([self.dirichlet_alpha] * self.action_space_size),
                    dtype=torch.float32,
                    device=self.device
                )
                
            root_policy = (1 - self.exploration_fraction) * root_policy + self.exploration_fraction * noise
        
        # Create root node
        root = BatchedNode(0)
        root.hidden_state = root_hidden_state
        
        # Expand root with the computed policy
        self._expand_node(root, valid_actions, root_policy)
        
        # Run simulations in batches
        for sim_idx in range(0, self.num_simulations, self.batch_size):
            # Adjusted batch size for the last batch
            current_batch_size = min(self.batch_size, self.num_simulations - sim_idx)
            
            # Prepare for batched simulation
            sim_nodes = []  # Track nodes for each simulation in batch
            sim_paths = []  # Track paths for each simulation
            
            # Select nodes for the batch
            for b in range(current_batch_size):
                # Selection phase - find leaf node
                node = root
                search_path = [node]
                
                # Traverse tree to find a leaf node (not yet expanded)
                while node.expanded():
                    action, node = self._select_child(node, valid_actions)
                    search_path.append(node)
                
                # Store the search path for this simulation
                sim_nodes.append(node)
                sim_paths.append(search_path)
            
            # Expansion phase - batch process leaf nodes
            # We're only expanding and evaluating nodes that haven't been expanded yet
            to_expand_indices = []
            parent_indices = []
            actions = []
            
            for i, (node, path) in enumerate(zip(sim_nodes, sim_paths)):
                if not node.expanded() and len(path) > 1:
                    to_expand_indices.append(i)
                    parent = path[-2]  # parent node
                    action = self._get_action(parent, node)
                    parent_indices.append(i)
                    actions.append(action)
            
            # If there are nodes to expand
            if len(to_expand_indices) > 0:
                # Prepare batches for network inference
                batch_size = len(to_expand_indices)
                hidden_batch = self.hidden_batch[:batch_size].clone()
                action_batch = self.action_batch[:batch_size].clone()
                
                # Fill the batches
                for i, (parent_idx, action) in enumerate(zip(parent_indices, actions)):
                    parent = sim_paths[parent_idx][-2]
                    hidden_batch[i] = parent.hidden_state[0]  # Assuming hidden_state is [1, hidden_dim]
                    action_batch[i] = action
                
                # Batched inference - all operations stay on GPU
                with torch.no_grad():
                    next_hidden_batch, reward_batch, value_batch, policy_logits_batch = (
                        self.network.recurrent_inference_batch(hidden_batch, action_batch)
                    )
                
                # Process the results and update nodes
                for i, expand_idx in enumerate(to_expand_indices):
                    node = sim_nodes[expand_idx]
                    
                    # Get data for this simulation
                    node.hidden_state = next_hidden_batch[i:i+1]  # Keep as [1, hidden_dim]
                    # Get reward as float but minimize host-device sync
                    with torch.no_grad():
                        node.reward = reward_batch[i].item()
                    
                    # Convert policy logits to probabilities - keep on device
                    policy = torch.softmax(policy_logits_batch[i:i+1], dim=1).squeeze(0)
                    
                    # Expand the node
                    self._expand_node(node, valid_actions, policy)
                    
                    # Backpropagate the value
                    # Use value from tensor but convert to Python float to avoid repeated sync
                    with torch.no_grad():
                        value = value_batch[i].item()
                    self._backpropagate(sim_paths[expand_idx], value)
            
            # For already expanded nodes (which can happen due to collisions in tree search)
            for i, (node, path) in enumerate(zip(sim_nodes, sim_paths)):
                if i not in to_expand_indices and node.expanded():
                    # Just backpropagate with the node's current value
                    value = node.value()
                    self._backpropagate(path, value)
        
        # Extract the search policy (normalized visit counts)
        search_policy = np.zeros(self.action_space_size, dtype=np.float32)
        
        # Update search policy based on visit counts
        visit_counts = np.array([root.children[a].visit_count if a in root.children else 0 
                                for a in range(self.action_space_size)])
        if visit_counts.sum() > 0:
            search_policy = visit_counts / visit_counts.sum()
            
        return search_policy
    
    def _expand_node(self, node: BatchedNode, valid_actions, policy):
        """
        Expand a node with the given policy.
        
        Args:
            node: The node to expand
            valid_actions: List of valid actions or None
            policy: The policy probabilities for each action
        """
        if valid_actions is not None:
            # Only expand valid actions
            for action in valid_actions:
                # Get policy value as float but minimize synchronization
                with torch.no_grad():
                    prior = policy[action].item()
                node.children[action] = BatchedNode(prior)
        else:
            # Expand all actions with non-zero prior
            for action in range(self.action_space_size):
                with torch.no_grad():
                    prior = policy[action].item()
                if prior > 0:
                    node.children[action] = BatchedNode(prior)
    
    def _select_child(self, node: BatchedNode, valid_actions=None):
        """
        Select the child with the highest UCB score.
        
        Args:
            node: The parent node
            valid_actions: List of valid actions or None
            
        Returns:
            (action, child_node): The selected action and child node
        """
        # Use valid actions if provided, otherwise use all children
        actions = valid_actions if valid_actions is not None else list(node.children.keys())
        
        # Filter to only actions that are children
        actions = [a for a in actions if a in node.children]
        
        # Calculate UCB scores
        ucb_scores = []
        for action in actions:
            child = node.children[action]
            
            # UCB formula components
            prior_score = child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            value_score = 0.0 if child.visit_count == 0 else child.value()
            
            # Complete UCB score
            ucb = prior_score + value_score
            ucb_scores.append(ucb)
        
        # Find action with highest UCB score
        max_index = np.argmax(ucb_scores)
        action = actions[max_index]
        return action, node.children[action]
    
    def _get_action(self, parent: BatchedNode, child: BatchedNode) -> int:
        """
        Get the action that leads from parent to child.
        
        Args:
            parent: The parent node
            child: The child node
            
        Returns:
            action: The action index
        """
        for action, node in parent.children.items():
            if node is child:
                return action
        # Should never happen if child is actually a child of parent
        raise ValueError("Child is not a child of parent")
    
    def _backpropagate(self, search_path: List[BatchedNode], value: float, gamma: float = None):
        """
        Backpropagate the value through the given path.
        
        Args:
            search_path: The path to backpropagate through
            value: The value to backpropagate
            gamma: Optional discount factor override
        """
        if gamma is None:
            gamma = self.discount
            
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            
            # No need to access node.reward which could be a tensor and cause synchronization
            # Instead we assume the value already includes the reward at this level
            
            # Apply discount for next level
            value = value * gamma


def run_batched_mcts(observation, network, valid_actions, num_simulations, device="cpu", batch_size=16):
    """
    Helper function to run batched MCTS in a single call.
    """
    mcts = BatchedMCTS(
        network=network,
        num_simulations=num_simulations,
        batch_size=batch_size,
        action_space_size=576,  # Narde-specific
        device=device
    )
    
    return mcts.run(observation, valid_actions, add_exploration_noise=True) 