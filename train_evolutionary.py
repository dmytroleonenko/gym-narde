import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import multiprocessing
from train_deepq_pytorch import DecomposedDQN  # reuse existing network architecture

# Hyperparameters for evolution
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
NUM_EPISODES_PER_INDIVIDUAL = 3
MUTATION_STD = 0.02
ELITE_FRACTION = 0.2

# Create evaluation environment (using your custom gym)
env = gym.make('Narde-v0', render_mode=None)

def simulate_game(model_white, model_black, max_steps=1000):
    """
    Simulate a single game between two candidate models.
    Candidate playing White uses model_white and candidate playing Black uses model_black.
    Both players act greedily (no exploration).
    Returns the winner: 1 if White wins, -1 if Black wins, or 0 for tie/undecided.
    """
    local_env = gym.make('Narde-v0', render_mode=None)
    state, _ = local_env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        if local_env.unwrapped.current_player == 1:
            # Use White candidate
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(model_white.device)
            with torch.no_grad():
                move1_q_values = model_white(state_tensor)
                best_move1 = torch.argmax(move1_q_values, dim=1).item()
                selected = torch.tensor([best_move1 % model_white.move_space_size], device=model_white.device)
                move2_q_values = model_white(state_tensor, selected)
                best_move2 = torch.argmax(move2_q_values, dim=1).item()
                action = (best_move1, best_move2)
        else:
            # Use Black candidate
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(model_black.device)
            with torch.no_grad():
                move1_q_values = model_black(state_tensor)
                best_move1 = torch.argmax(move1_q_values, dim=1).item()
                selected = torch.tensor([best_move1 % model_black.move_space_size], device=model_black.device)
                move2_q_values = model_black(state_tensor, selected)
                best_move2 = torch.argmax(move2_q_values, dim=1).item()
                action = (best_move1, best_move2)
        state, reward, done, truncated, _ = local_env.step(action)
        done = done or truncated
        steps += 1
    # Determine winner based on borne_off counts
    # We assume that reaching 15 borne off is a win.
    if local_env.unwrapped.game.borne_off_white == 15:
        winner = 1
    elif local_env.unwrapped.game.borne_off_black == 15:
        winner = -1
    else:
        winner = 0
    local_env.close()
    return winner

def simulate_match(args):
    """
    Simulate a round-robin match between two candidates.
    Args is a tuple: (i, j, model_i, model_j, num_games, max_steps)
    For each pairing, we play num_games with the roles swapped.
    Returns (i, wins_i, j, wins_j) representing the wins for each candidate.
    """
    i, j, model_i, model_j, num_games, max_steps = args
    wins_i = 0
    wins_j = 0
    # Round 1: candidate i plays White, j plays Black.
    for _ in range(num_games):
        result = simulate_game(model_white=model_i, model_black=model_j, max_steps=max_steps)
        if result == 1:
            wins_i += 1
        elif result == -1:
            wins_j += 1
    # Round 2: swap colors: candidate j is White, i is Black.
    for _ in range(num_games):
        result = simulate_game(model_white=model_j, model_black=model_i, max_steps=max_steps)
        if result == 1:
            wins_j += 1
        elif result == -1:
            wins_i += 1
    return (i, wins_i, j, wins_j)

def tournament_evaluation(population, num_games_per_match=5, max_steps=1000, parallel_games=8):
    """
    Run a full round-robin tournament among candidates in the population.
    Each pairing plays 2*num_games_per_match games (swapping colors).
    Returns a list of fitness values (total wins) for each candidate.
    """
    num_candidates = len(population)
    # Initialize wins for each candidate.
    wins = [0] * num_candidates
    
    # Create the list of match tasks for all unique pairs i < j.
    match_tasks = []
    for i in range(num_candidates):
        for j in range(i+1, num_candidates):
            match_tasks.append((i, j, population[i], population[j], num_games_per_match, max_steps))
    
    # Schedule matches in parallel using a Pool.
    with multiprocessing.Pool(processes=parallel_games) as pool:
        results = pool.map(simulate_match, match_tasks)
    
    # Accumulate wins from each match.
    for (i, wins_i, j, wins_j) in results:
        wins[i] += wins_i
        wins[j] += wins_j
    
    # Optionally, print tournament summary stats.
    total_matches = len(match_tasks)
    print(f"Tournament complete: {total_matches} match-ups played among {num_candidates} candidates.")
    for idx, w in enumerate(wins):
        print(f" Candidate {idx}: Total wins = {w}")
    
    return wins
    model.eval()
    total_reward = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        # For evolutionary evaluation, set a fixed max step count
        steps = 0
        while not done and steps < 1000:
            # For action selection, use the decomposed network greedy policy (no exploration)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(model.device)
            with torch.no_grad():
                move1_q_values = model(state_tensor)
                best_move1 = torch.argmax(move1_q_values, dim=1).item()
                # For second move, use best Q-value from second head with selected first move
                selected = torch.tensor([best_move1 % model.move_space_size], device=model.device)
                move2_q_values = model(state_tensor, selected)
                best_move2 = torch.argmax(move2_q_values, dim=1).item()
                action = (best_move1, best_move2)
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            ep_reward += reward
            steps += 1
        total_reward += ep_reward
    return total_reward / episodes

def mutate_model(model, std=MUTATION_STD):
    # Create a new model with perturbed parameters
    child = copy.deepcopy(model)
    for param in child.parameters():
        noise = torch.randn_like(param) * std
        param.data.add_(noise)
    return child

def main():
    # Create a dedicated output directory for saving models
    save_dir = os.path.join(os.getcwd(), "evolved_models")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    base_model = DecomposedDQN(state_size=28, move_space_size=576)
    base_model.to(torch.device("cpu"))
    # For evolution, we don't need an optimizer since we use random mutations
    population = [copy.deepcopy(base_model) for _ in range(POPULATION_SIZE)]
    
    best_fitness = -float("inf")
    best_model = None
    
    for gen in range(NUM_GENERATIONS):
        # Run a full round-robin tournament to compute win counts as fitness.
        fitnesses = tournament_evaluation(population,
                                          num_games_per_match=NUM_GAMES_PER_MATCH,
                                          max_steps=MAX_STEPS_MATCH,
                                          parallel_games=PARALLEL_GAMES)
        
        avg_fitness = np.mean(fitnesses)
        best_idx = np.argmax(fitnesses)
        gen_best = fitnesses[best_idx]
        print(f"Generation {gen+1} -- Avg fitness: {avg_fitness:.3f}, Best fitness: {gen_best:.3f}, Std Dev: {np.std(fitnesses):.3f}")

        # Save best model of current generation if it outperforms previous ones
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_model = copy.deepcopy(population[best_idx])
            torch.save(best_model.state_dict(), os.path.join(save_dir, f"evolved_model_gen{gen+1}.pt"))
            print(f"  New best model saved with fitness {best_fitness:.3f}")
        
        # Select elites
        num_elite = int(POPULATION_SIZE * ELITE_FRACTION)
        elite_indices = np.argsort(fitnesses)[-num_elite:]
        elites = [copy.deepcopy(population[idx]) for idx in elite_indices]
        
        # Breed new population: keep elites and fill remainder with mutated copies of elites
        new_population = elites.copy()
        while len(new_population) < POPULATION_SIZE:
            parent = np.random.choice(elites)
            child = mutate_model(parent)
            new_population.append(child)
        
        population = new_population

    # Save the final best model
    if best_model is not None:
        final_model_path = os.path.join(save_dir, "evolved_model_final.pt")
        torch.save(best_model.state_dict(), final_model_path)
        print(f"Final best model saved to {final_model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train an agent using evolutionary strategies for Narde')
    parser.add_argument('--population-size', type=int, default=50, help="Population size")
    parser.add_argument('--generations', type=int, default=100, help="Number of generations")
    parser.add_argument('--episodes-per-individual', type=int, default=3, help="Episodes per individual evaluation")
    parser.add_argument('--mutation-std', type=float, default=0.02, help="Standard deviation for mutation noise")
    parser.add_argument('--elite-fraction', type=float, default=0.2, help="Fraction of top-performing individuals to select as elites")
    
    parser.add_argument('--num-games', type=int, default=5, help="Number of games per match per color in tournament")
    parser.add_argument('--max-steps-match', type=int, default=1000, help="Max steps per game during tournament matches")
    parser.add_argument('--parallel-games', type=int, default=8, help="Number of games to run in parallel during tournament")
    
    args = parser.parse_args()
    
    # Override tournament hyperparameters with CLI parameters
    NUM_GAMES_PER_MATCH = args.num_games
    MAX_STEPS_MATCH = args.max_steps_match
    PARALLEL_GAMES = args.parallel_games
    
    # Override hyperparameters with CLI parameters
    POPULATION_SIZE = args.population_size
    NUM_GENERATIONS = args.generations
    NUM_EPISODES_PER_INDIVIDUAL = args.episodes_per_individual
    MUTATION_STD = args.mutation_std
    ELITE_FRACTION = args.elite_fraction
    
    main()
