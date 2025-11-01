import numpy as np
from scipy.ndimage import shift as ndimage_shift, map_coordinates
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Tuple, List

def generate_mock_bead_image(size: int, num_beads: int, noise_level: float = 0.05, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """Generates a mock image with random beads, applying optional uniform displacement."""
    img = np.zeros((size, size), dtype=np.float64)
    np.random.seed(42)  # For reproducible bead positions
    
    # Generate random bead centers and radii
    centers = []
    for _ in range(num_beads):
        # Center coordinates (x, y)
        r_center = np.random.randint(10, size - 10)
        c_center = np.random.randint(10, size - 10)
        centers.append((r_center, c_center))
        
    for r_center, c_center in centers:
        # Define bead parameters (for simple Gaussian-like spots)
        radius = np.random.uniform(2.0, 4.0)
        
        # Apply displacement to center for the deformed image
        r_current = r_center + dy
        c_current = c_center + dx
        
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

def bead_image_correlation(ref_img: np.ndarray, def_img: np.ndarray, subset_size: int, grid_spacing: int) -> np.ndarray:
    """
    Performs 2D Digital Image Correlation (DIC) to find the displacement field.
    
    Args:
        ref_img: The reference (undeformed) image.
        def_img: The deformed image.
        subset_size: The side length of the square subset (must be odd).
        grid_spacing: The distance between centers of adjacent subsets.
        
    Returns:
        A NumPy array of shape (N, 3), where N is the number of grid points.
        Each row is [y_coord, x_coord, u_displacement, v_displacement].
    """
    if subset_size % 2 == 0:
        raise ValueError("Subset size must be odd.")

    subset_half = subset_size // 2
    rows, cols = ref_img.shape
    
    # Define grid points (centers of subsets)
    # Start at subset_half and end before rows/cols - subset_half
    y_centers = np.arange(subset_half, rows - subset_half, grid_spacing)
    x_centers = np.arange(subset_half, cols - subset_half, grid_spacing)

    # List to store results: (y_center, x_center, u, v)
    displacement_field = []
    
    print(f"Starting DIC analysis with {len(y_centers) * len(x_centers)} subsets...")

    # Iterate over all grid points
    for r_center in y_centers:
        for c_center in x_centers:
            # 1. Extract the reference subset
            ref_subset = ref_img[r_center - subset_half : r_center + subset_half + 1,
                                 c_center - subset_half : c_center + subset_half + 1]

            # 2. Initial Guess for Displacement (u0, v0)
            # A simple initial guess of (0, 0) is used here. 
            initial_guess = np.array([0.0, 0.0]) # [u (x-disp), v (y-disp)]

            # Define bounds for the displacement search (+/- 5 pixels) for stability
            bounds = [(-5.0, 5.0), (-5.0, 5.0)]

            # 3. Sub-pixel Refinement using Optimization (Minimizing SSD)
            # Use 'COBYLA' as it supports bounds and is derivative-free, improving robustness.
            ref_center_tuple = (r_center, c_center)
            
            res = minimize(
                ssd_criterion, 
                initial_guess, 
                args=(ref_subset, def_img, ref_center_tuple, subset_half), 
                method='COBYLA', # Changed from 'Powell' to 'COBYLA'
                bounds=bounds,   # Added bounds for stability
                tol=1e-3,        # Set tolerance for convergence
                options={'maxiter': 500} # Increased maxiter for better convergence
            )
            
            if res.success:
                u, v = res.x
                # u is x-displacement (col), v is y-displacement (row)
                displacement_field.append([r_center, c_center, u, v])
            else:
                # If optimization fails, append NaN
                # print(f"Warning: Optimization failed at ({r_center}, {c_center}).")
                displacement_field.append([r_center, c_center, np.nan, np.nan])

    return np.array(displacement_field)

def plot_displacement_field(displacement_data: np.ndarray, image_size: int, applied_dx: float, applied_dy: float):
    """
    Plots the calculated displacement vectors.
    """
    # Remove NaN values (failed correlations)
    valid_data = displacement_data[~np.isnan(displacement_data).any(axis=1)]
    
    if valid_data.size == 0:
        print("No valid displacement data to plot.")
        return

    # Extract coordinates (Y, X) and displacements (U, V)
    Y, X = valid_data[:, 0], valid_data[:, 1]
    U, V = valid_data[:, 2], valid_data[:, 3]

    # Calculate magnitude for coloring
    Magnitude = np.sqrt(U**2 + V**2)
    
    # Calculate Mean Squared Error (MSE)
    # The true displacement applied was (applied_dx, applied_dy)
    error_u = U - applied_dx
    error_v = V - applied_dy
    mse = np.mean(error_u**2 + error_v**2)
    
    print("-" * 50)
    print(f"Mean Displacement U: {np.mean(U):.3f}")
    print(f"Mean Displacement V: {np.mean(V):.3f}")
    print(f"Applied Displacement: U={applied_dx}, V={applied_dy}")
    print(f"Mean Squared Error (MSE) in displacement: {mse:.4f} pixels^2")
    print("-" * 50)


    plt.figure(figsize=(8, 8), facecolor='#f8f8f8')
    # Quiver plot for the displacement field
    # The quiver function takes (X, Y, U, V)
    Q = plt.quiver(X, Y, U, V, 
                   Magnitude, 
                   pivot='mid', 
                   scale=1, 
                   scale_units='xy', 
                   angles='xy',
                   cmap='viridis')

    # Add color bar
    cbar = plt.colorbar(Q, label='Displacement Magnitude (pixels)')
    
    plt.title(f'DIC Displacement Field (Applied: $u={applied_dx}, v={applied_dy}$)', fontsize=14)
    plt.xlabel('X coordinate (pixels)', fontsize=12)
    plt.ylabel('Y coordinate (pixels)', fontsize=12)
    
    # Invert Y axis to match image coordinates (row=0 is top)
    plt.gca().invert_yaxis()
    plt.xlim(0, image_size)
    plt.ylim(image_size, 0)
    plt.axis('equal')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()