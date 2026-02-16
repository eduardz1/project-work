import networkx as nx

from tgp.types import SolutionType

EPSILON = 1e-4


def validate_solution(
    solution: SolutionType,
    G: nx.Graph,
    verbose: bool = True,
) -> tuple[bool, list[str]]:
    """Validate solution correctness: edges, depot start/end, weight tracking, gold collection."""
    errors = []

    if not solution:
        errors.append("Solution is empty")
        return False, errors

    for (c1, _), (c2, _) in zip(solution[:-1], solution[1:]):
        if not G.has_edge(c1, c2):
            errors.append(f"Invalid move from city {c1} to city {c2}: no edge in graph")

    if solution[0] != (0, 0.0):
        errors.append(f"Solution must start at (0, 0.0), got {solution[0]}")

    if solution[-1][0] != 0:
        errors.append(f"Solution must end at depot 0, got {solution[-1][0]}")

    gold_collected = {node: 0.0 for node in G.nodes}

    for i in range(len(solution) - 1):
        curr_city, curr_weight = solution[i]
        next_city, next_weight = solution[i + 1]

        if next_city != 0 and next_weight < curr_weight:
            errors.append(
                f"Step {i}: Weight decreased moving to city {next_city}: {curr_weight} → {next_weight}"
            )
            continue

        if next_city != 0 and curr_city != next_city:
            gold_picked_up = next_weight - curr_weight
            if gold_picked_up > 0:
                gold_collected[next_city] += gold_picked_up

    for node in G.nodes:
        if node == 0:
            continue
        expected_gold = G.nodes[node]["gold"]
        collected_gold = gold_collected[node]
        if abs(expected_gold - collected_gold) > EPSILON:
            errors.append(
                f"Node {node}: Expected {expected_gold:.2f} gold, got {collected_gold:.2f}"
            )

    if verbose and not errors:
        trips = sum(1 for city, _ in solution[1:] if city == 0)
        print(f"✓ Solution valid: {trips} trips, all gold collected")

    return not errors, errors
