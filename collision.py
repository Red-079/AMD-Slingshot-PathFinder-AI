import numpy as np


def compute_distances(sat_position, debris_positions):
    """
    Compute distances between satellite and all debris objects.
    """
    distances = np.linalg.norm(debris_positions - sat_position, axis=1)
    return distances


def detect_collision_risk(sat_position, debris_positions, threshold=50):
    """
    Check if any debris is within collision threshold.
    Returns:
        risk_detected (bool),
        closest_debris_position (array or None)
    """

    distances = compute_distances(sat_position, debris_positions)

    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    if min_distance < threshold:
        return True, debris_positions[min_index]

    return False, None


def estimate_collision_probability(distance, threshold=50):
    """
    Simple probability estimation based on distance.
    """
    if distance >= threshold:
        return 0.0

    probability = 1 - (distance / threshold)
    return round(probability, 3)

