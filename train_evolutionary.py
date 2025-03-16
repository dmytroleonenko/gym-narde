import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from train_deepq_pytorch import DecomposedDQN  # reuse existing network architecture

# Hyperparameters for evolution
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
NUM_EPISODES_PER_INDIVIDUAL = 3
MUTATION_STD = 0.02
ELITE_FRACTION = 0.2

# Create evaluation environment (using your custom gym)
env = gym.make('gym_narde:narde-v0', render_mode=None)

def evaluate_individual(model, episodes=NUM_EPISODES_PER_INDIVIDUAL):
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
    # Initialize base model
    base_model = DecomposedDQN(state_size=28, move_space_size=576)
    base_model.to(torch.device("cpu"))
    # For evolution, we don't need an optimizer since we use random mutations
    population = [copy.deepcopy(base_model) for _ in range(POPULATION_SIZE)]
    
    best_fitness = -float("inf")
    best_model = None
    
    for gen in range(NUM_GENERATIONS):
        fitnesses = []
        for i in range(POPULATION_SIZE):
            fitness = evaluate_individual(population[i])
            fitnesses.append(fitness)
        
        avg_fitness = np.mean(fitnesses)
        best_idx = np.argmax(fitnesses)
        gen_best = fitnesses[best_idx]
        print(f"Generation {gen+1} -- Avg fitness: {avg_fitness:.3f}, Best fitness: {gen_best:.3f}")

        # Save best model of current generation if it outperforms previous ones
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_model = copy.deepcopy(population[best_idx])
            torch.save(best_model.state_dict(), f"evolved_model_gen{gen+1}.pt")
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
        model_path = os.path.join(os.getcwd(), "evolved_model_final.pt")
        torch.save(best_model.state_dict(), model_path)
        print(f"Final best model saved to {model_path}")

if __name__ == "__main__":
    main()
