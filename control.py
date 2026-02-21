import numpy as np


# ==========================================================
# Delta-V Computation
# ==========================================================

def compute_delta_v(old_position, new_position):
    return np.linalg.norm(new_position - old_position)


# ==========================================================
# Fuel Consumption Model
# ==========================================================

def estimate_fuel_usage(delta_v, isp=300, mass=1000):
    g0 = 9.81
    fuel = mass * (1 - np.exp(-delta_v / (isp * g0)))
    return round(fuel, 4)


# ==========================================================
# PID Controller for Stabilization
# ==========================================================

class PIDController:

    def __init__(self, kp=0.1, ki=0.01, kd=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def compute(self, target, current):
        error = target - current
        self.integral += error
        derivative = error - self.previous_error

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.previous_error = error
        return output


# ==========================================================
# ADCS Maneuver Execution
# ==========================================================

def execute_maneuver(old_position, new_position):
    print("Executing ADCS maneuver...")
    print("Old Position:", old_position)
    print("Target Position:", new_position)

    delta_v = compute_delta_v(old_position, new_position)
    return new_position, delta_v


# ==========================================================
# Closed Loop Stabilization
# ==========================================================

def stabilize_position(target_position, current_position, steps=5):

    pid = PIDController()
    position = current_position.copy()

    for _ in range(steps):
        correction = pid.compute(target_position, position)
        position = position + correction

    return position


# ==========================================================
# Fault Tolerance Check
# ==========================================================

def system_health_check(position, max_radius=25000):

    radius = np.linalg.norm(position)

    if radius > max_radius:
        return False, "Orbit unstable"

    if radius < 100:
        return False, "Orbit too low"

    return True, "System stable"
