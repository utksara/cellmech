from typing import Annotated, Literal

import cv2
import numpy as np
import numpy.typing as npt
from scipy import interpolate
from scipy.ndimage import center_of_mass, label


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
    
    return gray, binary_output, centroids

def bead_image_correlation(image1 : np.ndarray, image2 : np.ndarray, bead_radius : float = 2, grid_res : int = 10):
    img1 = denoise_image(image1)
    img2 = denoise_image(image2)  
    _, _, centroids1 = cluster_beads(img1, bead_radius)
    _, _, centroids2 = cluster_beads(img2, bead_radius)
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



# Define a type alias for an (N, 2) array of floats
# The first dimension 'Any' represents N, and the second is explicitly 2
Vector2DArray = Annotated[npt.NDArray[np.float64], Literal["N", 2]]

def interpolate_vector_field(positions : Vector2DArray, values  : Vector2DArray, N : int = 100):
    xx, yy = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    u_interp = interpolate.griddata(positions, values[:, 1], (xx, yy), method='cubic')
    v_interp = interpolate.griddata(positions, values[:, 0], (xx, yy), method='cubic')
    return np.nan_to_num(np.stack([u_interp, v_interp], axis = -1), copy=False)

from scipy.spatial import KDTree

import numpy as np
from scipy.spatial import KDTree

def efficient_stable_map(points1, points2):
    n1, n2 = len(points1), len(points2)
    tree2 = KDTree(points2)
    tree1 = KDTree(points1)
    
    # State tracking
    p1_to_p2 = {}  # Final mapping: {index_p1: index_p2}
    p2_to_p1 = {}  # Reverse mapping: {index_p2: index_p1}
    
    # Track how many neighbors we've checked for each point in points1
    # We start by checking the 1st nearest neighbor (k=1)
    p1_search_k = np.ones(n1, dtype=int)
    unmatched_p1 = list(range(n1))

    while unmatched_p1:
        idx1 = unmatched_p1.pop(0)
        
        current_k = p1_search_k[idx1]
        if current_k > n2:
            continue # No more points left in points2 to check
            
        # Query the k-th nearest neighbor
        # We query 'k' neighbors and take the last one to get the k-th best
        distances, indices = tree2.query(points1[idx1], k=current_k)
        
        # If k > 1, query returns an array; if k == 1, it returns a scalar
        idx2 = indices[-1] if current_k > 1 else indices

        # Step 2: Check if idx1 is the 'best' (closest) for idx2
        # This is the "Stable Marriage" condition you requested
        _, best_p1_for_p2 = tree1.query(points2[idx2], k=1)
        
        if best_p1_for_p2 == idx1:
            # It's a mutual match (or idx1 is the preferred suitor)
            if idx2 in p2_to_p1:
                # If idx2 was already matched to someone else, 
                # that person is now unmatched and must look for their next neighbor
                old_p1 = p2_to_p1[idx2]
                unmatched_p1.append(old_p1)
                p1_search_k[old_p1] += 1 
            
            p1_to_p2[idx1] = idx2
            p2_to_p1[idx2] = idx1
        else:
            # idx2 prefers someone else. 
            # idx1 must move to its next closest neighbor in the next iteration
            p1_search_k[idx1] += 1
            unmatched_p1.append(idx1)

    return p1_to_p2

    
def maxmatch_difference(previous, after):
    mapping = efficient_stable_map(previous, after)
    p1_indices = list(mapping.keys())
    p2_indices = [mapping[idx] for idx in p1_indices]
    
    # Use NumPy indexing to get the coordinate subsets
    coords1 = previous[p1_indices]
    coords2 = after[p2_indices]
    
    # Vectorized subtraction: Displacement = Destination - Source
    displacements = coords2 - coords1
    
    return displacements, coords1

def clustered_bead_image_correlation(centroid_before: np.ndarray, centroid_after: np.ndarray, grid_res) -> np.ndarray:
    points_before = sort_beads(centroid_before)
    points_after = sort_beads(centroid_after)
    # points_before, points_after = augment_zeros_to_array(points_before, points_after)
    raw_displacements, positions = maxmatch_difference(points_before, points_after)
    # 5. Linear Interpolation (Delaunay-based)
    interp_vecfield = interpolate_vector_field(positions, raw_displacements, grid_res)
    return interp_vecfield
