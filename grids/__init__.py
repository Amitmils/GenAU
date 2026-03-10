import scipy
import torch
from .grid_creation import generate_sphere_points
import os

LEBEDEV_GRID_PATH = os.path.join(os.path.dirname(__file__),r"Lebvedev2702.mat")
LEBEDEV = "lebedev"
POINTS_162 = "162_points"

def create_grid(grid_type):
    if grid_type == LEBEDEV:
        num_grid_points = 2702  # P
        lebedev = scipy.io.loadmat(LEBEDEV_GRID_PATH)
        P_th = torch.tensor(lebedev["th"].reshape(-1))  # rad
        P_ph = torch.tensor(lebedev["ph"].reshape(-1))  # rad
        P_ph = (P_ph + torch.pi) % (2 * torch.pi) - torch.pi  # wrap angles to [-pi,pi]
    elif grid_type == POINTS_162:
        num_grid_points = 162  # P
        points = generate_sphere_points(162, plot=False)
        P_th = points[:, 1]
        P_ph = points[:, 2]
    else:
        raise ValueError(f"Unknown grid type {grid_type}")
    return P_th, P_ph, num_grid_points