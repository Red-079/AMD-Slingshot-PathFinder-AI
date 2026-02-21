import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime


# =====================================================
# CONFIGURATION
# =====================================================

SATELLITE_INITIAL_POSITION = np.array([100.0, 200.0, 300.0])
SATELLITE_VELOCITY = np.array([2.0, -1.5, 1.0])

DEBRIS_COUNT = 50
SPACE_LIMIT = 2000

COLLISION_THRESHOLD = 150
MAX_ORBIT_RADIUS = 25000


# =====================================================
# LOGGER
# =====================================================

def log(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


# =====================================================
# SIMULATION
# =====================================================

def generate_debris(n=DEBRIS_COUNT):
    return np.random.uniform(-SPACE_LIMIT, SPACE_LIMIT, (n, 3))


def propagate_orbit(position, velocity, dt=1.0):
    return position + velocity * dt


# =====================================================
# COLLISION ENGINE
# =====================================================

def detect_collision_risk(sat_position, debris_positions):
    distances = np.linalg.norm(debris_positions - sat_position, axis=1)
    min_index = np.argmin(distances)
    min_distance = distances[min_index]

    if min_distance < COLLISION_THRESHOLD:
        probability = 1 - (min_distance / COLLISION_THRESHOLD)
        return True, debris_positions[min_index], round(probability, 3)

    return False, None, 0.0


# =====================================================
# AI ENGINE
# =====================================================

class TrajectoryPredictor(nn.Module):
    def __init__(self):
        super(TrajectoryPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)


def predict_trajectory(model, position):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(position, dtype=torch.float32)
        predicted = model(input_tensor)
    return predicted.numpy()


def generate_maneuver(current_position, threat_position):
    avoidance_vector = current_position - threat_position
    direction = avoidance_vector / np.linalg.norm(avoidance_vector)
    delta_v = 20
    return current_position + direction * delta_v


def validate_maneuver(position):
    radius = np.linalg.norm(position)
    return radius < MAX_ORBIT_RADIUS


# =====================================================
# CONTROL SYSTEM
# =====================================================

def execute_maneuver(old_position, new_position):
    log("Executing ADCS maneuver...")
    return new_position


def closed_loop_correction(position):
    log("Applying stabilization...")
    return position * 0.99


def compute_fuel_usage(old_position, new_position):
    return round(np.linalg.norm(new_position - old_position), 3)


# =====================================================
# METRICS
# =====================================================

def compute_efficiency(fuel_used):
    baseline = 25
    if fuel_used == 0:
        return 1.0
    return round(baseline / fuel_used, 3)


# =====================================================
# VISUALIZATION
# =====================================================

def plot_environment(sat_position, debris_positions):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        debris_positions[:, 0],
        debris_positions[:, 1],
        debris_positions[:, 2],
        c='red',
        label='Debris'
    )

    ax.scatter(
        sat_position[0],
        sat_position[1],
        sat_position[2],
        c='blue',
        s=100,
        label='Satellite'
    )

    ax.set_title("Pathfinder AI Orbital Simulation")
    ax.legend()
    plt.show()


# =====================================================
# MAIN SYSTEM FLOW
# =====================================================

def main():

    log("Pathfinder AI Simulation Started")

    satellite_position = SATELLITE_INITIAL_POSITION.copy()
    debris_positions = generate_debris()

    # Orbit propagation
    satellite_position = propagate_orbit(satellite_position, SATELLITE_VELOCITY)

    log(f"Current Satellite Position: {satellite_position}")

    # Collision Detection
    risk_detected, threat_position, probability = detect_collision_risk(
        satellite_position,
        debris_positions
    )

    if not risk_detected:
        log("No collision risk detected.")
        plot_environment(satellite_position, debris_positions)
        return

    log(f"Collision Risk Detected. Probability: {probability}")

    # AI Prediction
    model = TrajectoryPredictor()
    predicted_position = predict_trajectory(model, satellite_position)
    log(f"Predicted Future Position: {predicted_position}")

    # Maneuver Planning
    new_position = generate_maneuver(satellite_position, threat_position)

    # Physics Validation
    if not validate_maneuver(new_position):
        log("Maneuver rejected due to physics constraint.")
        return

    log("Maneuver validated.")

    # Execution
    executed_position = execute_maneuver(satellite_position, new_position)
    stabilized_position = closed_loop_correction(executed_position)

    fuel_used = compute_fuel_usage(satellite_position, stabilized_position)
    efficiency = compute_efficiency(fuel_used)

    log(f"Fuel Used (Delta-V): {fuel_used}")
    log(f"Mission Efficiency Score: {efficiency}")

    plot_environment(stabilized_position, debris_positions)

    log("Simulation Completed Successfully")


if __name__ == "__main__":
    main()
