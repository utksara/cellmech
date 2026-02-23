from cellmech import detect_shapes, detect_shapes_canon
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import os

HANDRAWN_TOLERENCE = 5e-2
REALCELL_TOLERENCE = 1e-1

def assert_same_curve(points_a, points_b, rel_tolerance=0.01):
    """
    Asserts two point sets represent the same curve based on relative error.

    Parameters:
    -----------
    points_a, points_b : np.ndarray (m, 2)
    rel_tolerance : float
        Allowed error as a fraction of the curve's scale (e.g., 0.01 for 1%).
    """
    A = np.asanyarray(points_a)
    B = np.asanyarray(points_b)

    # 1. Compute the characteristic scale (Bounding Box Diagonal)
    # This represents the "size" of the curve
    def get_scale(pts):
        min_pts = np.min(pts, axis=0)
        max_pts = np.max(pts, axis=0)
        return np.linalg.norm(max_pts - min_pts)

    scale_a = get_scale(A)
    scale_b = get_scale(B)

    # Use the average scale of both curves to be fair
    avg_scale = (scale_a + scale_b) / 2.0

    # 2. Build KD-Trees
    tree_a = cKDTree(A)
    tree_b = cKDTree(B)

    # 3. Compute absolute Hausdorff distances
    dist_a_to_b, _ = tree_b.query(A, k=1)
    dist_b_to_a, _ = tree_a.query(B, k=1)

    abs_error = max(np.mean(dist_a_to_b), np.mean(dist_b_to_a))

    # 4. Calculate Relative Error
    # Handle the zero-scale case (point curves) to avoid division by zero
    if avg_scale < 1e-12:
        current_rel_error = abs_error
    else:
        current_rel_error = abs_error / avg_scale

    print(f"Absolute Max Deviation: {abs_error:.6e}")
    print(f"Curve Scale: {avg_scale:.6e}")
    print(f"Relative Error: {current_rel_error:.4%}")

    assert current_rel_error <= rel_tolerance, \
        f"Relative error {current_rel_error:.4%} exceeds tolerance {rel_tolerance:.4%}"

def test_handrawn_boundaries():
    for image in ['img1', 'img2', 'img3', 'img4']:
        image_matrix = np.array(Image.open(
            f'images/cell_boundary/{image}.png').convert('L'))
        shape_points_1, _ = detect_shapes(image_matrix)
        shape_points_2, _ = detect_shapes_canon(image_matrix)
        assert_same_curve(shape_points_1, shape_points_2, rel_tolerance=HANDRAWN_TOLERENCE)
        if int(os.environ.get("ENABLE_VISUAL_TESTING", False)):
            plt.plot(shape_points_1[:, 0], shape_points_1[:, 1])
            plt.plot(shape_points_2[:, 0], shape_points_2[:, 1])
            plt.show()

def test_real_cells():
    for image in ['img1', 'img4', 'img5']:
        image_matrix_origin = np.array(Image.open(
            f'images/real_cells/{image}.png').convert('L'))
        image_matrix_marked = np.array(Image.open(
            f'images/real_cells/{image}_marked.png').convert('L'))
        shape_points_1, _ = detect_shapes(image_matrix_origin)
        shape_points_2, _ = detect_shapes_canon(image_matrix_marked)
        assert_same_curve(shape_points_1, shape_points_2, rel_tolerance=REALCELL_TOLERENCE)
        if int(os.environ.get("ENABLE_VISUAL_TESTING", False)):
            plt.plot(shape_points_1[:, 0], shape_points_1[:, 1])
            plt.plot(shape_points_2[:, 0], shape_points_2[:, 1])
            plt.show()