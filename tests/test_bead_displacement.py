import os

import numpy as np
from PIL import Image

from cellmech import plot_vector_field
from cellmech.imgproc.bead_detection import bead_image_correlation

# Constants

def test_beads_displacement():    
    image_folder = "img4"
    reference_image = np.array(Image.open(f'images/beads/{image_folder}/before.png').convert('L'))
    deformed_image = np.array(Image.open(f'images/beads/{image_folder}/after.png').convert('L'))

    displacement_field = bead_image_correlation(
        reference_image, 
        deformed_image,
        grid_res=20
    )
    if int(os.environ.get("ENABLE_VISUAL_TESTING", False)):
        plot_vector_field(displacement_field)
