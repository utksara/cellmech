import numpy as np
from scipy.ndimage import shift as ndimage_shift, map_coordinates
from typing import Tuple
import cv2

def generate_mock_beads_center(num_beads: int, max_range: int):
    # Generate random bead centers and radii
    centers = []
    for _ in range(num_beads):
        # Center coordinates (x, y)
        r_center = np.random.randint(10, max_range)
        c_center = np.random.randint(10, max_range)
        centers.append((r_center, c_center))
    return centers

def get_fidelity(img1, img2, bead1, bead2):
    nx =  np.shape(img1)[0]
    ny =  np.shape(img1)[1]
    targt_sim1 = 1 - np.sum(abs(img1 - bead1))/(nx*ny)
    cross_sim1 = 1 - np.sum(abs(img1 - bead2))/(nx*ny)
    
    baseline_sim = 1 - np.sum(abs(bead1 - bead2))/(nx*ny)
    
    print(" baseline_sim ", baseline_sim)
    targt_sim2 = 1 - np.sum(abs(img2 - bead2))/(nx*ny)
    cross_sim2 = 1 - np.sum(abs(img2 - bead1))/(nx*ny)
    return targt_sim1 * (1 - cross_sim1) + targt_sim2 * (1 - cross_sim2)

def generate_mock_bead_image(size: int, centers: np.ndarray, noise_level: float = 0.05, dUx: np.ndarray = np.zeros((1, 1)), dUy: np.ndarray = np.zeros((1, 1))) -> np.ndarray:
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
    u, v = displacement[0], displacement[
        1]  # u is x-displacement (col), v is y-displacement (row)

    # Generate the local coordinate grid for the reference subset, centered at (0, 0)
    y_grid, x_grid = np.mgrid[-subset_half:subset_half +
                              1, -subset_half:subset_half+1]

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
        mode='nearest'  # Use nearest mode for boundary handling outside the check
    ).reshape(ref_subset.shape)

    # Calculate SSD
    ssd = np.sum((ref_subset - def_subset_interpolated)**2)
    return ssd

def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Denoises an image using OpenCV's Fast Non-Local Means Denoising.
    
    Args:
        image: np.ndarray (Grayscale or BGR).
        strength: Parameter 'h' regulating filter strength. 
                  Higher values remove more noise but blur details.
    """
    if image is None:
        raise ValueError("Input image is empty or None.")

    # Determine if the image is color or grayscale
    is_color = len(image.shape) == 3 and image.shape[2] == 3

    if is_color:
        # For colored images: 
        # h: filter strength, hColor: same for color components, 
        # templateWindowSize: 7, searchWindowSize: 21
        denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    else:
        # For grayscale images
        denoised = cv2.fastNlMeansDenoising(image, None, strength, 7, 21)

    return denoised

def kmeans_numpy(data, k, max_iters=10, tolerance=1.0):
    """
    Pure NumPy implementation of K-Means clustering.
    """
    # 1. Initialize centroids randomly from the data points
    n_samples = data.shape[0]
    if n_samples < k:
        return data # Not enough points to cluster
        
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[random_indices].astype(float)
    
    for _ in range(max_iters):
        # 2. Compute distances from each point to each centroid
        # Using broadcasting: (n_samples, 1, 2) - (1, k, 2) -> (n_samples, k, 2)
        # Then squared Euclidean distance
        distances = np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2, axis=2)
        
        # 3. Assign points to the nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # 4. Update centroids
        new_centroids = np.array([
            data[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])
        
        # 5. Check for convergence (if centroids move less than tolerance)
        center_shift = np.sum(np.sqrt(np.sum((new_centroids - centroids)**2, axis=1)))
        centroids = new_centroids
        
        if center_shift < tolerance:
            break
            
    return centroids

# function to convert raw bead image to well defined circular beads visible beads (grey sacle image with only binary pixel value)
def convert_to_bead_array(image: np.ndarray, num_beads: int = 100, bead_radius: int = 2) -> np.ndarray:
    """
    Converts an image to a binary 2D array of uniform 'beads' using 
    pure NumPy K-Means clustering.
    
    1 = Black (Bead), 0 = White (Background)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Invert image: dark regions become high values (more likely to attract beads)
    inv_gray = 255 - gray
    h, w = inv_gray.shape
    
    # 1. Weighted Sampling: Get coordinates based on pixel darkness
    # We create a coordinate grid and flatten it
    y_coords, x_coords = np.indices((h, w))
    coords = np.column_stack((y_coords.ravel(), x_coords.ravel()))
    
    # Use pixel intensities as weights for selection
    weights = inv_gray.ravel().astype(float)
    if weights.sum() == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    weights /= weights.sum()
    
    # Sample points to run K-Means on (sampling 5x the beads for a better distribution)
    sample_size = min(num_beads * 10, len(coords))
    sampled_indices = np.random.choice(len(coords), size=sample_size, p=weights, replace=False)
    sampled_coords = coords[sampled_indices]
    
    # 2. Run Pure NumPy K-Means
    # This finds the 'centers of gravity' for the dark regions
    centroids = kmeans_numpy(sampled_coords, num_beads)
    
    # 3. Create the Output 2D Array
    binary_output = np.zeros((h, w), dtype=np.uint8)
    
    for center in centroids:
        cy, cx = int(center[0]), int(center[1])
        # Draw a circle for each bead (ensures nearly identical area for all clusters)
        # Using cv2.circle is the most efficient way to modify the np.ndarray
        cv2.circle(binary_output, (cx, cy), bead_radius, 1, -1)
        
    return binary_output

def bead_image_correlation(ref_binary: np.ndarray, def_binary: np.ndarray, grid_res: int = 64) -> np.ndarray:
    L = ref_binary.shape[0]
    # 1. Extract Bead Centroids
    def get_beads(arr):
        _, _, _, centroids = cv2.connectedComponentsWithStats(arr.astype(np.uint8), connectivity=8)
        return centroids[1:] 

    pts_ref = get_beads(ref_binary) # Shape (M, 2)
    pts_def = get_beads(def_binary) # Shape (K, 2)

    if len(pts_ref) == 0 or len(pts_def) == 0:
        return np.zeros((grid_res, grid_res, 2))

    # 2. Nearest Neighbor Matching (Broadcasting)
    # Finds the closest bead in 'def' for every bead in 'ref'
    dist_matrix = np.sum((pts_ref[:, np.newaxis, :] - pts_def[np.newaxis, :, :])**2, axis=2)
    closest_indices = np.argmin(dist_matrix, axis=1)
    raw_displacements = pts_def[closest_indices] - pts_ref # Shape (M, 2)

    # 3. Create the N x N Grid
    grid_coords = np.linspace(0, L - 1, grid_res)
    grid_x, grid_y = np.meshgrid(grid_coords, grid_coords)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1) # Shape (N*N, 2)

    # 4. Inverse Distance Weighting (IDW) Interpolation
    # We calculate the influence of each bead on each grid point
    # To keep it efficient, we'll use the 8 nearest beads for each grid point
    displacement_field = np.zeros((grid_res * grid_res, 2))
    
    # Distance from every grid point to every bead
    # (N*N, M) distance matrix
    dists = np.sqrt(np.sum((grid_points[:, np.newaxis, :] - pts_ref[np.newaxis, :, :])**2, axis=2))
    
    # Avoid division by zero
    dists = np.maximum(dists, 1e-9)
    weights = 1.0 / (dists**2) # Inverse Square Law
    
    # Weighted average of displacements
    # (N*N, M) @ (M, 2) -> (N*N, 2)
    weighted_sum = np.sum(weights[:, :, np.newaxis] * raw_displacements[np.newaxis, :, :], axis=1)
    sum_of_weights = np.sum(weights, axis=1)[:, np.newaxis]
    
    displacement_field = weighted_sum / sum_of_weights

    # Reshape to (N, N, 2)
    # [..., 0] is dy (vertical), [..., 1] is dx (horizontal)
    res = displacement_field.reshape(grid_res, grid_res, 2)
    return res
   
def bead_image_correlation_deprecated(
    image1: np.ndarray,
    image2: np.ndarray,
    N: int,
    Nc: int = 8,
    min_corr: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-scale bead correlation:
    - Coarse displacement estimation with large windows
    - Interpolation to requested resolution

    Parameters
    ----------
    image1, image2 : 2D arrays
    N : int
        Requested output displacement grid size (N x N)
    Nc : int
        Coarse grid size (Nc x Nc), Nc << N
    min_corr : float
        Minimum correlation peak

    Returns
    -------
    Ux, Uy : np.ndarray
        Displacement fields of shape (N, N)
    """

    if image1.ndim != 2 or image2.ndim != 2:
        raise ValueError("Images must be 2D.")

    H = min(image1.shape[0], image2.shape[0])
    W = min(image1.shape[1], image2.shape[1])
    image1 = image1[:H, :W]
    image2 = image2[:H, :W]

    # --- Coarse grid window size ---
    wy = H // Nc
    wx = W // Nc
    w = min(wy, wx)

    Uc_x = np.full((Nc, Nc), np.nan)
    Uc_y = np.full((Nc, Nc), np.nan)

    win1d = np.hanning(w)
    window = np.outer(win1d, win1d)

    for j in range(Nc):
        for i in range(Nc):
            y0 = j * wy
            x0 = i * wx

            win1 = image1[y0:y0+w, x0:x0+w].astype(float)
            win2 = image2[y0:y0+w, x0:x0+w].astype(float)

            if win1.shape != (w, w) or win2.shape != (w, w):
                continue

            std1 = win1.std()
            std2 = win2.std()
            if std1 < 1e-6 or std2 < 1e-6:
                continue

            win1 = (win1 - win1.mean()) / std1
            win2 = (win2 - win2.mean()) / std2
            win1 *= window
            win2 *= window

            F1 = np.fft.fft2(win1)
            F2 = np.fft.fft2(win2)
            corr = np.fft.ifft2(F1 * np.conj(F2)).real
            corr = np.fft.fftshift(corr)

            peak = corr.max()
            if peak < min_corr:
                continue

            py, px = np.unravel_index(np.argmax(corr), corr.shape)
            dy = py - w // 2
            dx = px - w // 2

            Uc_x[j, i] = dx
            Uc_y[j, i] = dy

    # Replace NaNs with 0
    Uc_x = np.nan_to_num(Uc_x, nan=0.0)
    Uc_y = np.nan_to_num(Uc_y, nan=0.0)

    # --- Interpolate to N x N ---
    y_coarse = np.linspace(0, 1, Nc)
    x_coarse = np.linspace(0, 1, Nc)
    y_fine = np.linspace(0, 1, N)
    x_fine = np.linspace(0, 1, N)

    # Bilinear interpolation using separability
    Ux = np.zeros((N, N))
    Uy = np.zeros((N, N))

    for j in range(N):
        for i in range(N):
            yc = y_fine[j]
            xc = x_fine[i]

            jy = np.searchsorted(y_coarse, yc) - 1
            ix = np.searchsorted(x_coarse, xc) - 1
            jy = np.clip(jy, 0, Nc-2)
            ix = np.clip(ix, 0, Nc-2)

            y0 = y_coarse[jy]
            y1 = y_coarse[jy+1]
            x0 = x_coarse[ix]
            x1 = x_coarse[ix+1]

            ty = (yc - y0) / (y1 - y0 + 1e-12)
            tx = (xc - x0) / (x1 - x0 + 1e-12)

            # Bilinear interpolation
            Ux[j, i] = (
                (1-ty)*(1-tx)*Uc_x[jy, ix] +
                (1-ty)*tx*Uc_x[jy, ix+1] +
                ty*(1-tx)*Uc_x[jy+1, ix] +
                ty*tx*Uc_x[jy+1, ix+1]
            )
            Uy[j, i] = (
                (1-ty)*(1-tx)*Uc_y[jy, ix] +
                (1-ty)*tx*Uc_y[jy, ix+1] +
                ty*(1-tx)*Uc_y[jy+1, ix] +
                ty*tx*Uc_y[jy+1, ix+1]
            )

    return Ux, Uy
