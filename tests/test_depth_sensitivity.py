import os
from typing import Annotated, Literal
from PIL import Image
import numpy as np
import numpy.typing as npt

from cellmech import plot_vector_field, calculate_traction_below, calculate_traction_force, CellMechParameters
from cellmech.imgproc.bead_detection import bead_image_correlation
# Define a type alias for an (N, 2) array of floats
# The first dimension 'Any' represents N, and the second is explicitly 2
Vector2DArray = Annotated[npt.NDArray[np.float64], Literal["N", 2]]


def test_depth():
    image_folder = "img3"
    reference_image = np.array(Image.open(f'images/beads/{image_folder}/before.png').convert('L'))
    deformed_image = np.array(Image.open(f'images/beads/{image_folder}/after.png').convert('L'))

    displacement_field = bead_image_correlation(
        reference_image, 
        deformed_image,
        grid_res=20
    )
    
    # define all the parameters that are required to calculata traction forces
    cellmechparams = CellMechParameters({
        "p" : 0.5,
        "E" : 10e3,
        "pi" : np.pi,
        "N" : 50,
        "width" :2,
        "pixel_size" : 1e-7 
    })
    
    surface_traction = calculate_traction_force(displacement_field, cellmechparams)
    m = 5
    traction_beaneath = calculate_traction_below(surface_traction, depth=10, m = m, cellmechparams=cellmechparams)
    custom_scale = np.max(np.sqrt(surface_traction[:, :, 0]**2 + surface_traction[:,  :, 1]**2))
    if int(os.environ.get("ENABLE_VISUAL_TESTING", False)):
        for i in range(0, m):
            print(f" max at layer {i} : ", np.max(traction_beaneath[:, :, i, :]))
            plot_vector_field(traction_beaneath[:, :, i, :], title = f"traction at depth {i+1}", custom_scale=custom_scale)
