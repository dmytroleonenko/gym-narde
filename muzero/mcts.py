import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set


class Node:
    """
    Node class for Monte Carlo Tree Search.
    """
    def __init__(self, prior: float = 0.0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children: Dict[int, Node] = {}
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


class MCTS:
    """
    Monte Carlo Tree Search for MuZero.
    """
    def __init__(
        self,
        network,
        num_simulations: int,
        discount: float = 0.99,
        dirichlet_alpha: float = 0.3,
        exploration_fraction: float = 0.25,
        pb_c_base: int = 19652,
        pb_c_init: float = 1.25,
        action_space_size: int = 1152,
        device: str = "cpu"
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.discount = discount
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.action_space_size = action_space_size
        self.device = device
        
        # Cache frequently used tensors
        self._zeros_action_tensor = None
        self._action_mask_cache = {}
        
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
        
        # Get the latent state and predictions for the root node
        with torch.no_grad():
            root_hidden_state, root_value, root_policy_logits = self.network.initial_inference(observation.unsqueeze(0))
        
        # Create valid action mask
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
            masked_policy_logits = root_policy_logits.clone()
            masked_policy_logits[0, ~action_mask.bool()] = float('-inf')
            root_policy_logits = masked_policy_logits
            
        # Convert policy logits to probabilities
        root_policy = torch.softmax(root_policy_logits, dim=1).squeeze(0)
        
        # Create root node
        root = Node(0)
        root.hidden_state = root_hidden_state
        
        # Add exploration noise to the root node
        if add_exploration_noise:
            # Only apply noise to valid actions
            if valid_actions is not None:
                # Create or reuse the zeros tensor
                if self._zeros_action_tensor is None or self._zeros_action_tensor.device != self.device:
                    self._zeros_action_tensor = torch.zeros_like(root_policy)
                noise = self._zeros_action_tensor.clone()
                
                # Generate Dirichlet noise for valid actions
                valid_noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_actions))
                
                # Apply the noise to each valid action individually
                for i, action in enumerate(valid_actions):
                    noise[action] = valid_noise[i]
            else:
                noise = torch.FloatTensor(
                    np.random.dirichlet([self.dirichlet_alpha] * self.action_space_size)
                ).to(self.device)
                
            root_policy = (1 - self.exploration_fraction) * root_policy + self.exploration_fraction * noise
        
        # Expand root node with prior probabilities
        self.expand_node(root, valid_actions, root_policy)
        
        # Prepare tensors for batch inference during simulations
        batch_size = 1  # Start with single sample
        if hasattr(self.network, 'recurrent_inference_batch'):
            # Use batch inference if available
            batch_size = min(8, self.num_simulations)  # Use reasonable batch size
            
            # Pre-allocate buffers for batch inference
            hidden_batch = torch.zeros(batch_size, root_hidden_state.shape[1], device=self.device)
            action_batch = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            path_indices = []
            
            simulation_paths = []
            current_batch_size = 0
            
            # Run simulations with batched inference when possible
            for sim_idx in range(self.num_simulations):
                node = root
                search_path = [node]
                
                # Selection phase: Select the path to a leaf node
                while node.expanded():
                    action, node = self.select_child(node)
                    search_path.append(node)
                
                # Add to batch for expansion
                if current_batch_size < batch_size:
                    # Get parent node and action
                    parent = search_path[-2]
                    action = self.get_action(parent, node)
                    
                    # Add to batch
                    hidden_batch[current_batch_size] = parent.hidden_state
                    action_batch[current_batch_size] = action
                    path_indices.append(len(simulation_paths))
                    simulation_paths.append(search_path)
                    current_batch_size += 1
                
                # Process batch if full or on last simulation
                if current_batch_size == batch_size or sim_idx == self.num_simulations - 1:
                    if current_batch_size > 0:
                        # Process batch with network
                        with torch.no_grad():
                            next_hidden_batch, reward_batch, value_batch, policy_logits_batch = (
                                self.network.recurrent_inference_batch(
                                    hidden_batch[:current_batch_size],
                                    action_batch[:current_batch_size]
                                )
                            )
                        
                        # Process results for each path
                        for batch_idx, path_idx in enumerate(path_indices):
                            path = simulation_paths[path_idx]
                            node = path[-1]
                            
                            # Store node information
                            node.hidden_state = next_hidden_batch[batch_idx:batch_idx+1]
                            node.reward = reward_batch[batch_idx].item()
                            
                            # Convert policy logits to probabilities
                            policy = torch.softmax(policy_logits_batch[batch_idx:batch_idx+1], dim=1).squeeze(0)
                            
                            # Expand the node with prior probabilities
                            self.expand_node(node, valid_actions, policy)
                            
                            # Backpropagate
                            self.backpropagate(path, value_batch[batch_idx].item(), self.discount)
                        
                        # Reset for next batch
                        current_batch_size = 0
                        path_indices.clear()
                        simulation_paths.clear()
        else:
            # Fallback to sequential simulations
            for _ in range(self.num_simulations):
                node = root
                search_path = [node]
                
                # Selection phase: Select the path to a leaf node
                while node.expanded():
                    action, node = self.select_child(node)
                    search_path.append(node)
                
                # Get parent node and the action that led to the selected node
                parent = search_path[-2]
                action = self.get_action(parent, node)
                
                # Expansion phase: Expand the leaf node using network
                with torch.no_grad():
                    hidden_state, reward, value, policy_logits = self.network.recurrent_inference(
                        parent.hidden_state, 
                        torch.tensor([action], device=self.device)
                    )
                
                # Convert policy logits to probabilities
                policy = torch.softmax(policy_logits, dim=1).squeeze(0)
                
                # Store node information
                node.hidden_state = hidden_state
                node.reward = reward.item()
                
                # Expand the node with prior probabilities
                self.expand_node(node, valid_actions, policy)
                
                # Backup phase: Update the statistics of all nodes in the search path
                self.backpropagate(search_path, value.item(), self.discount)
        
        # Extract the search policy (visit counts)
        # Pre-allocate the array with zeros
        search_policy = np.zeros(self.action_space_size, dtype=np.float32)
        
        # Update only the actions that have children
        for action, child in root.children.items():
            search_policy[action] = child.visit_count
        
        # Normalize the policy
        search_policy_sum = np.sum(search_policy)
        if search_policy_sum > 0:
            search_policy /= search_policy_sum
            
        return search_policy
    
    def expand_node(self, node, valid_actions, policy):
        """
        Expand a node with the given policy.
        
        Args:
            node: The node to expand
            valid_actions: List of valid actions
            policy: The policy probabilities for each action
        """
        # Only consider valid actions
        actions_to_expand = valid_actions if valid_actions is not None else range(self.action_space_size)
        
        # Pre-allocate dictionary capacity
        if not node.children and len(actions_to_expand) > 0:
            node.children = {}
        
        for action in actions_to_expand:
            # Create a new child node with the prior probability
            node.children[action] = Node(prior=policy[action].item())
    
    def select_child(self, node):
        """
        Select a child according to the PUCT formula.
        
        Args:
            node: The node to select a child from
            
        Returns:
            A tuple (action, child)
        """
        # Calculate UCB score for each child more efficiently
        best_score = float('-inf')
        best_action = -1
        best_child = None
        
        # Cache the PUCT formula constants
        pb_c = math.log((node.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c_mult = math.sqrt(node.visit_count)
        
        for action, child in node.children.items():
            # PUCT formula
            prior_score = pb_c * child.prior * pb_c_mult / (child.visit_count + 1)
            value_score = child.value()
            ucb_score = prior_score + value_score
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def get_action(self, parent, child):
        """
        Get the action that led from parent to child.
        
        Args:
            parent: The parent node
            child: The child node
            
        Returns:
            The action
        """
        for action, node in parent.children.items():
            if node is child:
                return action
        raise ValueError("Child not found in parent's children")
    
    def backpropagate(self, search_path, value, discount):
        """
        Backpropagate the value through the search path.
        
        Args:
            search_path: List of nodes forming the search path
            value: The value to backpropagate
            discount: The discount factor
        """
        # Start from the leaf node and work backwards
        for i in range(len(search_path) - 1, -1, -1):
            node = search_path[i]
            node.value_sum += value
            node.visit_count += 1
            
            # For the next iteration, include the reward from the current node
            # and apply the discount
            if i > 0:  # Skip the root node (no incoming reward)
                value = search_path[i].reward + discount * value 