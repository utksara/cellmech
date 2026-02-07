from cellmech.imgproc.bead_detection import bead_image_correlation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
# Constants
SUBSET_SIZE = 31  # Must be odd
GRID_SPACING = 5
grid_size = 200
num_beads = 100
IMAGE_SIZE = grid_size

reference_image = np.array(Image.open('images/img5/bead_image_before.png').convert('L'))
deformed_image = np.array(Image.open('images/img5/bead_image_after.png').convert('L'))

# 3. Perform DIC
# try:
displacement_field_array = bead_image_correlation(
    reference_image, 
    deformed_image, grid_size
)

# Extract coordinates (Y, X) and displacements (U, V)
# Y, X = displacement_field_array[:, 0], displacement_field_array[:, 1]
U, V = displacement_field_array
sns.heatmap(U**2 + V**2)
plt.title("displacement using bead corr")
plt.legend()
plt.show()
    