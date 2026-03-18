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

        test_field = np.zeros((5,5,2))
        test_field[:,:,0] = np.flip( np.array([
            [0, 0,  0, 0, 0],
            [0, 1, -0.5, 0, 0],
            [0, 0,  0,   0, 0],
            [0, 0,  0,   0, 0],
            [0, 0,  0,   0, 0]
            ]).T, axis = 1)

        test_field[:,:,1] = np.flip( np.array([
            [0, 0,     0,   0, 0],
            [0, -0.1, -0.1, 0, 0],
            [0,    1, -0.9, 0, 0],
            [0,    0,  0,   0, 0],
            [0,    0,  0,   0, 0]
            ]).T, axis = 1)

            
        print(test_field[2,3,0])
        print(test_field[2,2,1])
        fig = plot_vector_field(test_field, "vector map of tensor displacement field")