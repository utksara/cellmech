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

def bead_image_correlation(image1, image2, N):
    """
    Calculates the displacement field using Thin Plate Spline (TPS) interpolation.
    Maintains the requested signature and internal logic flow.
    """
    
    # 1. Initialize NxN displacement field
    dUx = np.zeros((N, N))
    dUy = np.zeros((N, N))
    
    # 2. Extract bead positions (finding local maxima)
    def get_bead_positions(img, threshold_percentile=98):
        threshold = np.percentile(img, threshold_percentile)
        data_max = maximum_filter(img, size=5)
        maxima = (img == data_max) & (img > threshold)
        return np.argwhere(maxima) # Returns [row, col] -> [y, x]

    beads1 = get_bead_positions(image1)
    beads2 = get_bead_positions(image2)

    # 3. Map beads and calculate displacement (O(M) correlation)
    ux_list, uy_list, coords_list = [], [], []

    for b1 in beads1:
        # Simple Euclidean distance mapping to find the same bead in image2
        distances = np.linalg.norm(beads2 - b1, axis=1)
        if len(distances) == 0: continue
        
        idx = np.argmin(distances)
        if distances[idx] < 30:  # Search radius threshold
            b2 = beads2[idx]
            uy_list.append(b2[0] - b1[0])
            ux_list.append(b2[1] - b1[1])
            coords_list.append(b1)

    # Convert to arrays: P contains [y, x], U contains [uy, ux]
    P = np.array(coords_list)
    U = np.vstack([uy_list, ux_list]).T
    M = P.shape[0]

    if M < 3:
        raise ValueError("Not enough beads found to calculate a 2D displacement field.")

    # 4. TPS Interpolation Logic (More accurate for few beads)
    # Radial Basis Function kernel: r^2 * log(r)
    def rbf(r):
        mask = r > 0
        res = np.zeros_like(r)
        res[mask] = r[mask]**2 * np.log(r[mask])
        return res

    # Build the linear system L * W = Y
    # K is (M x M) distance matrix through RBF
    dist_matrix = np.linalg.norm(P[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2)
    K = rbf(dist_matrix)
    
    # Q is (M x 3) matrix containing [1, y, x]
    Q = np.hstack([np.ones((M, 1)), P])
    
    # Construct the full TPS matrix
    L = np.block([
        [K,               Q],
        [Q.T, np.zeros((3, 3))]
    ])
    
    # Targeted values (displacements) padded with zeros for the affine part
    Y_x = np.concatenate([U[:, 1], np.zeros(3)])
    Y_y = np.concatenate([U[:, 0], np.zeros(3)])
    
    # Solve for weights (coefficients)
    weights_x = np.linalg.solve(L, Y_x)
    weights_y = np.linalg.solve(L, Y_y)

    # 5. Compute entire displacement field on NxN grid
    rows, cols = image1.shape
    y_grid = np.linspace(0, rows - 1, N)
    x_grid = np.linspace(0, cols - 1, N)
    grid_Y, grid_X = np.meshgrid(y_grid, x_grid, indexing='ij')
    
    # Flatten grid for vectorized computation
    flat_grid = np.stack([grid_Y.ravel(), grid_X.ravel()], axis=1)
    
    # Distance from every grid point to every bead
    dist_grid_bead = np.linalg.norm(flat_grid[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2)
    K_grid = rbf(dist_grid_bead)
    Q_grid = np.hstack([np.ones((N*N, 1)), flat_grid])
    
    # Final displacement calculation: (RBF Part) + (Affine Part)
    dUx = (K_grid @ weights_x[:M]) + (Q_grid @ weights_x[M:])
    dUy = (K_grid @ weights_y[:M]) + (Q_grid @ weights_y[M:])

    return dUx.reshape(N, N), dUy.reshape(N, N)
    
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
