from cellmech import plot_vector_field
import numpy as np
import os
from scipy import interpolate
from numpy.random import rand

from typing import Annotated, Literal
import numpy.typing as npt

# Define a type alias for an (N, 2) array of floats
# The first dimension 'Any' represents N, and the second is explicitly 2
Vector2DArray = Annotated[npt.NDArray[np.float64], Literal["N", 2]]


def interpolate_vector_field(positions : Vector2DArray, values  : Vector2DArray, N : int = 100):
    xx, yy = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    u_interp = interpolate.griddata(positions, values[:, 0], (xx, yy), method='linear')
    v_interp = interpolate.griddata(positions, values[:, 1], (xx, yy), method='linear')
    return np.nan_to_num(np.stack([u_interp, v_interp], axis = -1), copy=False)

def test_interpolation():
    N = 20
    points = np.array([[0.25, 0.5], [0.5, 0.75], [0.75, 0.5], [0.5, 0.25]])
    values = np.array([[1, 0], [0,-0.5], [-0.5, 0], [0, 0.25]])
    displacement_field = interpolate_vector_field(points, values, N)
    print(displacement_field.shape)
    if int(os.environ.get("ENABLE_VISUAL_TESTING", False)):
        plot_vector_field(displacement_field)
