import numpy as np
from scipy.ndimage import shift as ndimage_shift, map_coordinates
from scipy.interpolate import LinearNDInterpolator
from typing import Tuple
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

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
        print("Not enough points to cluster!")
        return data
        
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


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Invert image: dark regions become high values (more likely to attract beads)    
    dark = gray < np.mean(gray)
    ligt = gray >= np.mean(gray)
    if  np.sum(dark) < np.sum(ligt):
        gray = 255 - gray
        
    gray[gray < np.mean(gray)] = 0
    return gray

from scipy.ndimage import label, center_of_mass

def find_bead_coordinates(grayscale_image: np.ndarray):
    """
    Identifies bead clusters from a grayscale image using a 3x3 window 
    vectorized filter and returns their original coordinates.
    """
    M, N = grayscale_image.shape
    
    # 1. Normalize the image
    img_max = np.max(grayscale_image)
    if img_max == 0:
        return np.array([])
    norm_img = grayscale_image / img_max
    
    # 2. Determine m and n (number of 3x3 blocks)
    m, n = M // 3, N // 3
    
    # 3. Vectorized operation: 
    # Extract the M x N area that fits the 3x3 windows perfectly
    trimmed_img = norm_img[:3*m, :3*n]
    
    # Reshape into (m, 3, n, 3) and sum over the 3x3 windows (axes 1 and 3)
    # This simulates applying the [[1,1,1],[1,1,1],[1,1,1]] filter
    windows_sum = trimmed_img.reshape(m, 3, n, 3).sum(axis=(1, 3))
    
    # 4. Create the binary m x n array based on threshold
    threshold = 5
    binary_grid = (windows_sum > threshold).astype(int)
    
    # 5. Clustering: Identify distinct groups of ones
    # structure=np.ones((3,3)) allows for diagonal connectivity
    labeled_array, num_clusters = label(binary_grid)
    
    if num_clusters == 0:
        return np.empty((0, 2), dtype=int)
    
    # 6. Pinpoint positions (Centroids)
    # Get center of mass of clusters in the (m, n) space
    cluster_indices = np.arange(1, num_clusters + 1)
    centroids = center_of_mass(binary_grid, labeled_array, cluster_indices)
    centroids = np.array(centroids)
    
    # 7. Map back to original M x N coordinates
    # Since each point in binary_grid represents a 3x3 block, 
    # we multiply by 3 to scale back and add 1.5 to hit the center of the block.
    original_coords = centroids * 3 + 1.5
    
    # Ensure coordinates are within [0, M) and [0, N)
    original_coords[:, 0] = np.clip(original_coords[:, 0], 0, M - 1)
    original_coords[:, 1] = np.clip(original_coords[:, 1], 0, N - 1)
    
    original_coords[:, 0] = original_coords[:,0]/M
    original_coords[:, 1] = original_coords[:,1]/N
    return original_coords
    # return np.flip(original_coords, axis = 1)
    
# function to convert raw bead image to well defined circular beads visible beads (grey sacle image with only binary pixel value)
def cluster_beads(image: np.ndarray, bead_radius: int = 5) -> np.ndarray:
    """
    Converts an image to a binary 2D array of uniform 'beads' using 
    pure NumPy K-Means clustering.
    
    """
    gray = convert_to_grayscale(image) 
     
    h, w = gray.shape
    centroids = find_bead_coordinates(gray)
    binary_output = np.zeros((h, w), dtype=np.uint8)
    
    for center in centroids:
        cy, cx = int(center[0]), int(center[1])
        # Draw a circle for each bead (ensures nearly identical area for all clusters)
        cv2.circle(binary_output, (cx, cy), bead_radius, 1, -1)
    
    # sns.heatmap(binary_output)
    # plt.title(" New grayscale image")
    # plt.show()
    return gray, binary_output, centroids

def bead_image_correlation(image1 : np.ndarray, image2 : np.ndarray, bead_radius : float = 2, grid_res : int = 10):
    img1 = denoise_image(image1)
    img2 = denoise_image(image2)  
    _, _, centroids1 = cluster_beads(img1, bead_radius)
    _, _, centroids2 = cluster_beads(img2, bead_radius)
    print("centroids1 - \n", centroids1)
    print("centroids2 - \n", centroids2)
    return clustered_bead_image_correlation(centroids1, centroids2, grid_res)
    
    
def augment_zeros_to_array(a, b):
    if a.shape[0] < b.shape[0]: a = np.pad(a, ((0, b.shape[0] - a.shape[0]), (0, 0)))
    elif b.shape[0] < a.shape[0]: b = np.pad(b, ((0, a.shape[0] - b.shape[0]), (0, 0)))
    return a, b

def sort_beads(arr):
    # # 1. Pure geometric centroid extraction
    # _, _, _, centroids = cv2.connectedComponentsWithStats(arr.astype(np.uint8), connectivity=8)
    # pts = centroids[1:] # Exclude background
    # # 2. Force deterministic ordering by coordinates (Y then X)
    return arr[np.lexsort((arr[:, 0], arr[:, 1]))]

from typing import Annotated, Literal
import numpy.typing as npt
from scipy import interpolate
from numpy.random import rand

# Define a type alias for an (N, 2) array of floats
# The first dimension 'Any' represents N, and the second is explicitly 2
Vector2DArray = Annotated[npt.NDArray[np.float64], Literal["N", 2]]

def interpolate_vector_field(positions : Vector2DArray, values  : Vector2DArray, N : int = 100):
    xx, yy = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    u_interp = interpolate.griddata(positions, values[:, 0], (xx, yy), method='cubic')
    v_interp = interpolate.griddata(positions, values[:, 1], (xx, yy), method='cubic')
    return np.nan_to_num(np.stack([u_interp, v_interp], axis = -1), copy=False)
    
def clustered_bead_image_correlation(ref_centroid: np.ndarray, def_centroid: np.ndarray, grid_res) -> np.ndarray:
    pts_ref = sort_beads(ref_centroid)
    pts_def = sort_beads(def_centroid)
    pts_ref, pts_def = augment_zeros_to_array(pts_ref, pts_def)
    raw_displacements = pts_def - pts_ref
    print("raw displacements : ", raw_displacements)
    print("initial positions : ", pts_ref)
    # 5. Linear Interpolation (Delaunay-based)
    interp_vecfield = interpolate_vector_field(pts_ref, raw_displacements, grid_res)
    print("interpolated disp : ", interp_vecfield[:,:,0])
    return interp_vecfield
   
def bead_image_correlation_deprecated(
    image1: np.ndarray,
    image2: np.ndarray,
    N : int = 100,
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
    return np.stack([Ux, Uy], axis=2)
