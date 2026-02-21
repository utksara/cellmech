from cellmech.imgproc.bead_detection import bead_image_correlation
from cellmech import plot_vector_field
import numpy as np
from PIL import Image
import os
# Constants

def test_beads_displacement():    
    reference_image = np.array(Image.open('images/beads/img7/before.png').convert('L'))
    deformed_image = np.array(Image.open('images/beads/img7/after.png').convert('L'))

    displacement_field = bead_image_correlation(
        reference_image, 
        deformed_image, 10, min_corr = 0.04
    )
    Ux, Uy = displacement_field
    if os.environ.get("ENABLE_VISUAL_TESTING", False):
        plot_vector_field(np.stack([Ux, Uy], axis = 2))
