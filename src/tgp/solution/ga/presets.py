from tgp.solution.ga.crossovers import iox
from tgp.solution.ga.mutations import inversion_mutation, swap_mutation

PRESETS = {
    "fast": {
        "population_size_percent": 0.23,
        "generations_percent": 0.21,
        "elitism_rate": 0.17,
        "mutation_rate": 0.04,
        "lns_rate": 0.04,
        "lns_num_to_remove_percent": 0.25,
        "tournament_size_percent": 0.14,
        "crossover": iox,
        "mutation": inversion_mutation,
    },
    "balanced": {
        "population_size_percent": 0.23,
        "generations_percent": 0.37,
        "elitism_rate": 0.11,
        "mutation_rate": 0.01,
        "lns_rate": 0.48,
        "lns_num_to_remove_percent": 0.24,
        "tournament_size_percent": 0.14,
        "crossover": iox,
        "mutation": swap_mutation,
    },
    "quality": {
        "population_size_percent": 0.51,
        "generations_percent": 0.86,
        "elitism_rate": 0.32,
        "mutation_rate": 0.04,
        "lns_rate": 0.48,
        "lns_num_to_remove_percent": 0.24,
        "tournament_size_percent": 0.14,
        "crossover": iox,
        "mutation": inversion_mutation,
    },
}
