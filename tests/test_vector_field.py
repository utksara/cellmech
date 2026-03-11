import os

import numpy as np

from cellmech import plot_vector_field

# Constants

def test_vector_map():
    N = 20
    grid = np.linspace(-1, 1, N)  
    X, Y = np.meshgrid(grid, grid)
    sigm1 = 0.5
    sigm2 = 2
    displacement_field = np.zeros((N, N, 2))
    displacement_field[:, :, 0] = np.exp(-X**2/(2* sigm1)) * np.exp(-Y**2/(2* sigm1))
    # displacement_field[:, :, 1] = np.exp(-X**2/(2* sigm2)) * np.exp(-Y**2/(2* sigm2))
    if int(os.environ.get("ENABLE_VISUAL_TESTING", False)):
        plot_vector_field(displacement_field)
