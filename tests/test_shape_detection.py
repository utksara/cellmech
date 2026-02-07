from cellmech import detect_shapes
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns

if __name__ == "__main__":
    # Constants
    SUBSET_SIZE = 31  # Must be odd
    GRID_SPACING = 20
    grid_size = 100
    num_beads = 10
    image_matrix = np.array(Image.open(
        'images/cell_boundary/img5.png').convert('L'))
    force_points, updated_image = detect_shapes(image_matrix)
    print(len(force_points))
    sns.heatmap(updated_image)
    plt.show()
