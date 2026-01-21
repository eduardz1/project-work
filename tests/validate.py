import networkx as nx

from tgp.types import SolutionType

EPSILON = 1e-4


def validate_solution(
    solution: SolutionType,
    G: nx.Graph,
    verbose: bool = True,
) -> tuple[bool, list[str]]:
    """Validate solution correctness: depot start/end, weight tracking, gold collection."""
    errors = []

    if not solution:
        errors.append("Solution is empty")
        return False, errors

    if solution[0][0] != 0 or solution[0][1] != 0.0:
        errors.append(f"Solution must start at (0, 0.0), got {solution[0]}")

    if solution[-1][0] != 0:
        errors.append(f"Solution must end at depot 0, got {solution[-1][0]}")

    gold_collected = {node: 0.0 for node in G.nodes}

    for i in range(len(solution) - 1):
        curr_city, curr_weight = solution[i]
        next_city, next_weight = solution[i + 1]

        if curr_city == 0 and next_city == 0:
            if next_weight > curr_weight:
                errors.append(
                    f"Step {i}: Weight increased at depot: {curr_weight} → {next_weight}"
                )
            continue

        if curr_city not in G.nodes:
            errors.append(f"Step {i}: Unknown node {curr_city}")
        if next_city not in G.nodes:
            errors.append(f"Step {i}: Unknown node {next_city}")

        if curr_city == next_city:
            errors.append(f"Step {i}: No movement (stayed at city {curr_city})")
            continue

        weight_diff = next_weight - curr_weight
        if next_city == 0:
            if abs(weight_diff) > EPSILON:
                errors.append(
                    f"Step {i}: Weight changed returning to depot: {curr_weight} → {next_weight}"
                )
        else:
            gold_at_dest = G.nodes[next_city]["gold"]
            if abs(weight_diff) < EPSILON:
                pass
            elif abs(weight_diff - gold_at_dest) < EPSILON:
                gold_collected[next_city] += gold_at_dest
            elif weight_diff > 0:
                gold_collected[next_city] += weight_diff
            else:
                errors.append(
                    f"Step {i}: Weight decreased moving to city {next_city}: {curr_weight} → {next_weight}"
                )

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
