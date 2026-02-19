import numpy as np
import torch

from ai_engine import TrajectoryPredictor, predict_trajectory, generate_maneuver, validate_maneuver
from collision import detect_collision_risk, compute_distances, estimate_collision_probability
from control import execute_maneuver, closed_loop_correction, compute_fuel_usage


# -----------------------------
# 1. Simulation Setup
# -----------------------------
def generate_debris(n=20):
    return np.random.uniform(-1000, 1000, (n, 3))


def propagate_orbit(position, velocity, dt=1):
    return position + velocity * dt


# -----------------------------
# 2. Main Execution Flow
# -----------------------------
def main():

    print("\n--- Pathfinder AI Simulation Started ---\n")

    # Initialize satellite state
    satellite_position = np.array([100.0, 200.0, 300.0])
    satellite_velocity = np.array([2.0, -1.5, 1.0])

    debris_positions = generate_debris()

    # Propagate satellite orbit
    satellite_position = propagate_orbit(satellite_position, satellite_velocity)

    print("Current Satellite Position:", satellite_position)

    # -----------------------------
    # Collision Detection
    # -----------------------------
    risk_detected, threat_position = detect_collision_risk(
        satellite_position,
        debris_positions,
        threshold=150
    )

    if not risk_detected:
        print("No collision risk detected.")
        return

    print("Collision Risk Detected.")
    distance = np.linalg.norm(threat_position - satellite_position)
    probability = estimate_collision_probability(distance, threshold=150)

    print("Collision Probability:", probability)

    # -----------------------------
    # AI Trajectory Prediction
    # -----------------------------
    model = TrajectoryPredictor()
    predicted_position = predict_trajectory(model, satellite_position)

    print("Predicted Next Position:", predicted_position)

    # -----------------------------
    # Maneuver Planning
    # -----------------------------
    new_position = generate_maneuver(satellite_position, threat_position)

    # -----------------------------
    # Physics Validation
    # -----------------------------
    is_valid = validate_maneuver(new_position)

    if not is_valid:
        print("Maneuver rejected due to physics constraints.")
        return

    print("Maneuver validated.")

    # -----------------------------
    # Autonomous Execution
    # -----------------------------
    executed_position = execute_maneuver(satellite_position, new_position)

    # Closed-loop stabilization
    stabilized_position = closed_loop_correction(executed_position)

    fuel_used = compute_fuel_usage(satellite_position, stabilized_position)

    print("Fuel Used (Delta-V approximation):", fuel_used)

    print("\n--- Simulation Completed Successfully ---\n")


if __name__ == "__main__":
    main()

