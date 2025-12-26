import numpy as np
from scipy.ndimage import shift as ndimage_shift, map_coordinates
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
from typing import Tuple, List
from PIL import Image
from cellmech.utils import symmetric_gaussian
from cellmech.fttc import detect_shapes

def generate_mock_beads_center(num_beads: int, max_range : int):
    # Generate random bead centers and radii
    centers = []
    for _ in range(num_beads):
        # Center coordinates (x, y)
        r_center = np.random.randint(10, max_range)
        c_center = np.random.randint(10, max_range)
        centers.append((r_center, c_center))
    return centers
    
def generate_mock_bead_image(size: int, centers: np.ndarray, noise_level: float = 0.05, dUx : np.ndarray = np.zeros((1,1)), dUy: np.ndarray = np.zeros((1,1))) -> np.ndarray:
    """Generates a mock image with random beads, applying optional uniform displacement."""
    img = np.zeros((size, size), dtype=np.float64)
    np.random.seed(42)  # For reproducible bead positions
        
    for r_center, c_center in centers:
        # Define bead parameters (for simple Gaussian-like spots)
        radius = np.random.uniform(2.0, 4.0)
        
        # Apply displacement to center for the deformed image
        r_current = r_center + dUx[r_center, c_center] 
        c_current = c_center + dUy[r_center, c_center]
        
        # Create a Gaussian spot (approximating a bead)
        for r in range(size):
            for c in range(size):
                distance_sq = (r - r_current)**2 + (c - c_current)**2
                intensity = np.exp(-distance_sq / (2.0 * radius**2))
                img[r, c] += intensity
                
    # Normalize and add noise
    img = (img - img.min()) / (img.max() - img.min())
    noise = np.random.randn(size, size) * noise_level
    img += noise
    img = np.clip(img, 0, 1)
    
    return img

def ssd_criterion(displacement: Tuple[float, float], ref_subset: np.ndarray, def_img: np.ndarray, ref_center: Tuple[int, int], subset_half: int) -> float:
    """
    Sum of Squared Differences (SSD) objective function for optimization.
    
    This function calculates the SSD between the reference subset and the
    interpolated deformed subset shifted by the candidate displacement (u, v).
    
    Args:
        displacement (u, v): Candidate displacement in x (columns) and y (rows).
        ref_subset: The reference image subset (SUBSET_SIZE x SUBSET_SIZE).
        def_img: The full deformed image.
        ref_center: (row, col) coordinates of the subset center in the reference image.
        subset_half: Half the subset size (SUBSET_SIZE // 2).
        
    Returns:
        The SSD value (a scalar to be minimized).
    """
    u, v = displacement[0], displacement[1] # u is x-displacement (col), v is y-displacement (row)
    
    # Generate the local coordinate grid for the reference subset, centered at (0, 0)
    y_grid, x_grid = np.mgrid[-subset_half:subset_half+1, -subset_half:subset_half+1]
    
    # Calculate the floating point coordinates in the deformed image space (Y, X)
    # Coordinates = (Reference Center + Displacement) + Local Grid Offset
    y_def = y_grid + ref_center[0] + v
    x_def = x_grid + ref_center[1] + u
    
    # Check if the coordinates are within bounds. If not, penalize heavily.
    if np.any(y_def < 0) or np.any(y_def >= def_img.shape[0]) or \
       np.any(x_def < 0) or np.any(x_def >= def_img.shape[1]):
        # Return a very large value to discourage this displacement near the boundary
        return 1e10
    
    # The coordinates array for map_coordinates must be [y_coords, x_coords] and flattened
    coords = np.array([y_def.flatten(), x_def.flatten()])
    
    # Interpolate the deformed subset G_d(x+u, y+v) using map_coordinates
    # Order=3 for cubic spline interpolation (high quality).
    def_subset_interpolated = map_coordinates(
        input=def_img, 
        coordinates=coords, 
        order=3, 
        mode='nearest' # Use nearest mode for boundary handling outside the check
    ).reshape(ref_subset.shape)

    # Calculate SSD
    ssd = np.sum((ref_subset - def_subset_interpolated)**2)
    return ssd

import numpy as np
from scipy.ndimage import maximum_filter

import numpy as np
from scipy.ndimage import maximum_filter

def bead_image_correlation(image1, image2, N):
    """
    Calculates a smooth, non-vanishing displacement field using 
    Normalized Gaussian Radial Basis Function interpolation.
    """
    
    # 1. Initialize NxN displacement field
    dUx = np.zeros((N, N))
    dUy = np.zeros((N, N))
    
    # 2. Extract bead positions (Local Maxima)
    def get_bead_positions(img):
        # Threshold at 95th percentile to isolate bright beads
        thresh = np.percentile(img, 95)
        local_max = maximum_filter(img, size=5) == img
        coords = np.argwhere(local_max & (img > thresh))
        return coords # [y, x]

    beads1 = get_bead_positions(image1)
    beads2 = get_bead_positions(image2)

    if len(beads1) == 0 or len(beads2) == 0:
        return dUx, dUy

    # 3. Map beads (Nearest Neighbor)
    ux_list, uy_list, pos_list = [], [], []
    for b1 in beads1:
        # Vectorized distance to find corresponding bead in image2
        dist = np.linalg.norm(beads2 - b1, axis=1)
        idx = np.argmin(dist)
        if dist[idx] < 25: # Max search radius in pixels
            b2 = beads2[idx]
            uy_list.append(b2[0] - b1[0])
            ux_list.append(b2[1] - b1[1])
            pos_list.append(b1)

    Ux = np.array(ux_list)
    Uy = np.array(uy_list)
    P = np.array(pos_list) # [y, x]
    M = P.shape[0]

    # 4. Numerically efficient grid computation
    # Determine an optimal sigma (width) based on average bead spacing
    # This prevents the field from "vanishing" between beads.
    sigma = 20.0 # Standard width; could be dynamic: np.mean(dist_to_neighbors)

    rows, cols = image1.shape
    y_grid = np.linspace(0, rows - 1, N)
    x_grid = np.linspace(0, cols - 1, N)
    grid_Y, grid_X = np.meshgrid(y_grid, x_grid, indexing='ij')
    
    # Flatten for vectorized broadcasting
    flat_Y = grid_Y.ravel()
    flat_X = grid_X.ravel()

    # Process in batches if memory is an issue, but for NxN it's usually fine
    # Calculate squared distances: (N*N, M)
    # dist^2 = (y_grid - y_bead)^2 + (x_grid - x_bead)^2
    dy = flat_Y[:, np.newaxis] - P[:, 0]
    dx = flat_X[:, np.newaxis] - P[:, 1]
    dist_sq = dy**2 + dx**2

    # 5. Gaussian Summation with Normalization (The Fix)
    # weights(i,j) = exp(-dist^2 / (2*sigma^2))
    weights = np.exp(-dist_sq / (2 * sigma**2))
    
    # sum_weights is the total influence at each grid point
    sum_weights = np.sum(weights, axis=1)
    
    # Avoid division by zero in empty areas
    sum_weights[sum_weights == 0] = 1e-9

    # dUx(i,j) = sum(Ux_bead * weight) / sum(weights)
    flat_dUx = np.sum(Ux * weights, axis=1) / sum_weights
    flat_dUy = np.sum(Uy * weights, axis=1) / sum_weights

    return flat_dUx.reshape(N, N), flat_dUy.reshape(N, N)
    
def generate_mock_displacement(image_file : str = 'images/cell_boundary/img5.png', N = 100, width = 1): 
    image_matrix = np.array(Image.open(image_file).convert('L'))
    force_points, updated_image = detect_shapes(image_matrix, detection_threshold=0.5)
    dx = width/N
    dim = width/2
    X = np.linspace(-dim, dim, N)
    U = np.zeros((N, N, 2))
    for point in force_points:
        dU = np.zeros((N, N, 2))
        v2 = np.array(point)
        for i in range(0, N):
            for j in range(0, N):
                v1 = np.array((X[i], X[j]))
                dU[i, j, :] = symmetric_gaussian(v1 - v2)  * dx * dx
        U += dU
    return U[:, :, 0], U[:, :, 1]
