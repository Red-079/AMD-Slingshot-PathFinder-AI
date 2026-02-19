import numpy as np
import torch
import torch.nn as nn

# 1. Trajectory Prediction Model

class TrajectoryPredictor(nn.Module):
    """
    Simplified Transformer-style trajectory predictor.
    Takes current position as input and predicts next position.
    """

    def __init__(self, input_dim=3, hidden_dim=32, output_dim=3):
        super(TrajectoryPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def predict_trajectory(model, current_position):
    """
    Predict next orbital position using AI model.
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(current_position, dtype=torch.float32)
        predicted = model(input_tensor)
    return predicted.numpy()


# 2. Reinforcement Learning Maneuver Logic

def generate_maneuver(current_position, threat_vector):
    """
    Simple RL-style maneuver logic.
    Moves spacecraft away from threat direction.
    """

    avoidance_direction = current_position - threat_vector
    normalized_direction = avoidance_direction / np.linalg.norm(avoidance_direction)

    delta_v = 10  # Mock fuel impulse magnitude
    new_position = current_position + normalized_direction * delta_v

    return new_position

# 3. Physics-Informed Validation

def validate_maneuver(position, max_orbit_radius=20000):
    """
    Ensures maneuver stays within allowed orbital constraints.
    """

    radius = np.linalg.norm(position)

    if radius > max_orbit_radius:
        return False

    return True

