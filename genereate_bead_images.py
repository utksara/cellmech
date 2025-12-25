from cellmech.bead_detection import bead_image_correlation, generate_mock_beads_center, generate_mock_bead_image, generate_mock_displacement
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
SUBSET_SIZE = 31  # Must be odd
GRID_SPACING = 5
grid_size = 200
num_beads = 100
IMAGE_SIZE = grid_size

# 1. Define Applied Displacement
APPLIED_DX = 2.5  # True displacement in X (columns)
APPLIED_DY = 1.8  # True displacement in Y (rows)

# # 3. Perform DIC
# # try:
# displacement_field_array = bead_image_correlation(
#     reference_image, 
#     deformed_image, 
#     subset_size=SUBSET_SIZE, 
#     grid_spacing=GRID_SPACING
# )
# # Extract coordinates (Y, X) and displacements (U, V)
# # Y, X = displacement_field_array[:, 0], displacement_field_array[:, 1]
# U, V = displacement_field_array

for image_name in ["img2", "img3", "img4", "img5", "img6"]:
    
    # 2. Generate Mock Images (Ref and Def)
    print(f"Generating mock bead image for {image_name}...")
        
    dUx, dUy = generate_mock_displacement(image_file = f'images/cell_boundary/{image_name}.png', N = grid_size)
    dUx = 1e4*dUx
    dUy = 1e4*dUy
    bead_centers = generate_mock_beads_center(num_beads=num_beads, max_range = grid_size)
    reference_image = generate_mock_bead_image(IMAGE_SIZE, bead_centers, noise_level=0.02, dUx=np.zeros((IMAGE_SIZE, IMAGE_SIZE)), dUy=np.zeros((IMAGE_SIZE, IMAGE_SIZE)))
    deformed_image = generate_mock_bead_image(IMAGE_SIZE, bead_centers, noise_level=0.02, dUx=dUx, dUy=dUy)
    print("Image generation complete.")
    os.makedirs(f'images/{image_name}', exist_ok=True)
    
    filename1 = f'images/{image_name}/bead_image_before.png'
    filename2 = f'images/{image_name}/bead_image_after.png'
    
    plt.imsave(filename1, reference_image, cmap='gray')
    plt.imsave(filename2, deformed_image, cmap='gray')