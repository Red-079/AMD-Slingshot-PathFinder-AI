import numpy as np


def execute_maneuver(current_position, new_position):
    """
    Simulate ADCS actuator execution.
    """
    print("Executing maneuver...")
    print("Old Position:", current_position)
    print("New Position:", new_position)

    return new_position


def closed_loop_correction(position, correction_factor=0.98):
    """
    Simulates feedback stabilization after maneuver.
    """
    stabilized_position = position * correction_factor
    print("Applying closed-loop stabilization...")
    return stabilized_position


def compute_fuel_usage(old_position, new_position):
    """
    Estimate Delta-V (fuel usage approximation).
    """
    delta_v = np.linalg.norm(new_position - old_position)
    return round(delta_v, 3)

