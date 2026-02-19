import numpy as np


def generate_debris(n=20, space_limit=1000):
    """
    Generate random debris positions in 3D space.
    
    Parameters:
        n (int): Number of debris objects
        space_limit (int): Maximum coordinate range
    
    Returns:
        numpy array of shape (n, 3)
    """
    debris_positions = np.random.uniform(-space_limit, space_limit, (n, 3))
    return debris_positions


def initialize_satellite():
    """
    Initialize satellite position and velocity.
    
    Returns:
        position (numpy array)
        velocity (numpy array)
    """
    position = np.array([100.0, 200.0, 300.0])
    velocity = np.array([2.0, -1.5, 1.0])
    return position, velocity


def propagate_orbit(position, velocity, dt=1.0):
    """
    Propagate satellite orbit using simple linear motion.
    
    Parameters:
        position (numpy array): Current satellite position
        velocity (numpy array): Satellite velocity vector
        dt (float): Time step
    
    Returns:
        Updated position
    """
    new_position = position + velocity * dt
    return new_position


def update_environment(position, velocity, debris_positions, dt=1.0):
    """
    Simulate one time step of orbital environment.
    
    Returns:
        Updated satellite position
        Updated debris positions
    """
    # Update satellite
    new_sat_position = propagate_orbit(position, velocity, dt)

    # Simulate slight random drift in debris
    debris_drift = np.random.uniform(-1, 1, debris_positions.shape)
    new_debris_positions = debris_positions + debris_drift

    return new_sat_position, new_debris_positions

