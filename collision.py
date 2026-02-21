import numpy as np


# ==========================================================
# Distance Computation
# ==========================================================

def compute_distances(sat_position, debris_positions):
    return np.linalg.norm(debris_positions - sat_position, axis=1)


# ==========================================================
# Time-To-Collision Estimation
# ==========================================================

def estimate_time_to_collision(sat_position, sat_velocity,
                               debris_position, debris_velocity):
    relative_position = debris_position - sat_position
    relative_velocity = debris_velocity - sat_velocity

    relative_speed = np.linalg.norm(relative_velocity)

    if relative_speed == 0:
        return np.inf

    time_to_collision = np.dot(relative_position, relative_velocity) / (relative_speed ** 2)

    return max(time_to_collision, 0)


# ==========================================================
# Collision Probability Model
# ==========================================================

def estimate_collision_probability(distance, relative_speed,
                                   threshold=150):
    distance_factor = max(0, 1 - (distance / threshold))
    velocity_factor = min(relative_speed / 10, 1)

    probability = 0.6 * distance_factor + 0.4 * velocity_factor
    return round(probability, 3)


# ==========================================================
# Conjunction Analysis (Multi-Debris Ranking)
# ==========================================================

def analyze_conjunctions(sat_position, sat_velocity,
                         debris_positions, debris_velocities):

    threat_list = []

    for i in range(len(debris_positions)):

        debris_pos = debris_positions[i]
        debris_vel = debris_velocities[i]

        distance = np.linalg.norm(debris_pos - sat_position)
        relative_speed = np.linalg.norm(debris_vel - sat_velocity)

        probability = estimate_collision_probability(distance, relative_speed)
        ttc = estimate_time_to_collision(
            sat_position, sat_velocity,
            debris_pos, debris_vel
        )

        threat_list.append({
            "index": i,
            "distance": distance,
            "probability": probability,
            "time_to_collision": ttc
        })

    threat_list.sort(key=lambda x: x["probability"], reverse=True)

    return threat_list


# ==========================================================
# Risk Detection
# ==========================================================

def detect_highest_risk(threat_list, probability_threshold=0.3):

    if not threat_list:
        return False, None

    highest = threat_list[0]

    if highest["probability"] > probability_threshold:
        return True, highest

    return False, None
