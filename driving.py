import math
import random
import networkx as nx
import numpy as np
from scipy import stats
from tqdm.auto import tqdm


def distance(coord1, coord2):
    """haversine formula - outputs miles"""
    R = 3958.8  # Earth radius in miles
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance_miles = R * c
    driving_distance = distance_miles * 1.4 # 1.4 accounts for road-length. see https://circuity.org/
    return driving_distance


def distance_with_uncertainty(coord1, coord2, uncertainty_factor=0.1):
    """Calculate distance and add uncertainty"""
    base_distance = distance(coord1, coord2)
    
    # Sample from a normal distribution centered at base_distance
    sampled_distance = np.random.normal(base_distance, base_distance * uncertainty_factor)
    return max(0, sampled_distance)  # Ensure non-negative distance

#  calculate the total length of a tour
def tour_length(cities, tour):
    return sum(distance(cities[tour[i]], cities[tour[i+1]]) for i in range(len(tour) - 1))

def tour_length_with_uncertainty(cities, tour):
    return sum(distance_with_uncertainty(cities[tour[i]], cities[tour[i+1]]) for i in range(len(tour) - 1))


def nearest_neighbor(cities):
    n = len(cities)
    visited = [False] * n
    tour = [0]
    visited[0] = True

    for _ in range(n - 1):
        last = tour[-1]
        nearest_city = None
        nearest_distance = float('inf')
        
        for i in range(n):
            if not visited[i] and distance(cities[last], cities[i]) < nearest_distance:
                nearest_city = i
                nearest_distance = distance(cities[last], cities[i])
        
        tour.append(nearest_city)
        visited[nearest_city] = True

    tour.append(0)  # Return to the starting city
    return tour


def mst_approximation(cities):
    G = nx.Graph()
    
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            G.add_edge(i, j, weight=distance(cities[i], cities[j]))
    
    T = nx.minimum_spanning_tree(G)
    pre_order = list(nx.dfs_preorder_nodes(T, 0))
    pre_order.append(0)  # Return to the starting city
    return pre_order


def simulated_annealing(cities, initial_temp, cooling_rate, seed=None):
    if seed is not None:
        random.seed(seed)
    
    current_tour = list(range(len(cities)))
    current_distance = tour_length(cities, current_tour)
    
    temp = initial_temp
    while temp > 1:
        new_tour = current_tour[:]
        i, j = random.sample(range(len(cities)), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        
        new_distance = tour_length(cities, new_tour)
        
        if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temp):
            current_tour = new_tour
            current_distance = new_distance
        
        temp *= cooling_rate
    
    current_tour.append(current_tour[0])  # Return to the starting city
    return current_tour


def genetic_algorithm(cities, population_size, generations, mutation_rate):
    population = [random.sample(range(len(cities)), len(cities)) for _ in range(population_size)]
    for individual in population:
        individual.append(individual[0])
    
    for _ in range(generations):
        population = sorted(population, key=lambda ind: tour_length(cities, ind))
        new_population = population[:population_size // 2]
        
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:population_size // 2], 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    population = sorted(population, key=lambda ind: tour_length(cities, ind))
    return population


def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1) - 1), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]
    
    fill = [item for item in parent2 if item not in child]
    fill_index = 0
    for i in range(len(child) - 1):
        if child[i] is None:
            child[i] = fill[fill_index]
            fill_index += 1
    
    child[-1] = child[0]
    return child


def mutate(individual):
    i, j = random.sample(range(len(individual) - 1), 2)
    individual[i], individual[j] = individual[j], individual[i]
    return individual


def monte_carlo_simulation(cities, num_simulations):
    solutions = []
    for _ in range(num_simulations):
        tour = random.sample(range(len(cities)), len(cities))
        tour.append(tour[0])
        solutions.append(tour)
    return solutions


def simulated_annealing_with_uncertainty(cities, initial_temp, cooling_rate, num_iterations=100):
    results = []
    
    for _ in tqdm(range(num_iterations)):
        current_tour = list(range(len(cities)))
        current_distance = tour_length_with_uncertainty(cities, current_tour)
        
        temp = initial_temp
        while temp > 1:
            new_tour = current_tour[:]
            i, j = random.sample(range(len(cities)), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            
            new_distance = tour_length_with_uncertainty(cities, new_tour)
            
            if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temp):
                current_tour = new_tour
                current_distance = new_distance
            
            temp *= cooling_rate
        
        current_tour.append(current_tour[0])  # Return to the starting city
        results.append((current_tour, current_distance))
    
    return results


def analyze_results(results):
    distances = [result[1] for result in results]
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    ci = stats.t.interval(df=len(distances)-1, loc=mean_distance, scale=stats.sem(distances), confidence=0.95)
    
    return {
        'mean': mean_distance,
        'std': std_distance,
        'ci_lower': ci[0],
        'ci_upper': ci[1]
    }


if __name__ == "__main__":
    cities_coordinates = [
        
        (37.5407246, -77.4360481),   # Richmond
        (38.9338676, -77.1772604),    # McLean
        
        (33.4483771, -112.0740373),  # Phoenix
        (39.7392358, -104.990251),   # Denver
        (40.7607793, -111.8910474),  # Salt Lake City
        # (30.267153, -97.7430608),    # Austin

        (40.7127837, -74.0059413),   # New York
        (42.3600825, -71.0588801),   # Boston
        (36.1626638, -86.7816016),   # Nashville, TN
        (43.661471, -70.2553259),    # Portland, Maine

        (47.6062095, -122.3320708),  # Seattle
        (34.0522342, -118.2436849),  # Los Angeles
        
        # (32.715738, -117.1610838),   # San Diego
        # (41.8781136, -87.6297982),   # Chicago
    ]
    
    nn_tour = nearest_neighbor(cities_coordinates)
    print(f"Nearest Neighbor: Tour = {nn_tour}, Length = {tour_length(cities_coordinates, nn_tour)}")
    
    mst_tour = mst_approximation(cities_coordinates)
    print(f"MST Approximation: Tour = {mst_tour}, Length = {tour_length(cities_coordinates, mst_tour)}")
    
    sa_tour = simulated_annealing(cities_coordinates, initial_temp=10000, cooling_rate=0.995)
    print(f"Simulated Annealing: Tour = {sa_tour}, Length = {tour_length(cities_coordinates, sa_tour)}")
    
    ga_population = genetic_algorithm(cities_coordinates, population_size=10, generations=100, mutation_rate=0.01)
    for i, ga_tour in enumerate(ga_population[:5]):
        print(f"Genetic Algorithm Solution {i+1}: Tour = {ga_tour}, Length = {tour_length(cities_coordinates, ga_tour)}")
    
    mc_solutions = monte_carlo_simulation(cities_coordinates, num_simulations=100)
    for i, mc_tour in enumerate(mc_solutions[:5]):
        print(f"Monte Carlo Solution {i+1}: Tour = {mc_tour}, Length = {tour_length(cities_coordinates, mc_tour)}")
    
    print("\nSimulated Annealing with Uncertainty:")
    results = simulated_annealing_with_uncertainty(cities_coordinates, initial_temp=10000, cooling_rate=0.995, num_iterations=100)
    analysis = analyze_results(results)
    
    print(f"Mean tour length: {analysis['mean']:.2f} miles")
    print(f"Standard deviation: {analysis['std']:.2f} miles")
    print(f"95% Confidence Interval: ({analysis['ci_lower']:.2f}, {analysis['ci_upper']:.2f}) miles")
    
    best_tour, best_distance = min(results, key=lambda x: x[1])
    print(f"\nBest tour found: {best_tour}")
    print(f"Best tour distance: {best_distance:.2f} miles")
