import cv2
import matplotlib.pyplot as plt
import numpy as np


def smooth_contours(contour):
    return smooth_contour_moving_average(contour, window_size = int(len(contour)/20) )

def smooth_contour_moving_average(contour, window_size):
    """
    contour: Nx1x2 or Nx2 array
    window_size: number of neighboring points to average (odd number recommended)
    """
    n = len(contour)
    contour = np.array(contour)
    # Ensure shape Nx2
    pts = contour.reshape(-1, 2)

    # Make contour circular
    pad = window_size // 2
    pts_padded = np.vstack([pts[-pad:], pts, pts[:pad]])

    kernel = np.ones(window_size) / window_size

    x_smooth = np.convolve(pts_padded[:, 0], kernel, mode='valid')
    y_smooth = np.convolve(pts_padded[:, 1], kernel, mode='valid')

    smoothed = np.stack([x_smooth[:-1], y_smooth[:-1]], axis=1)

    return smoothed.reshape(-1, 2)


# task : modify this function as specified
def detect_shapes(image_matrix: np.ndarray, detection_threshold=0.2):

    # Example improvements
    blur = cv2.GaussianBlur(image_matrix, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_matrix = clahe.apply(blur)
    # 1. Improve contrast

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(image_matrix)

    # 2. Edge detection
    upper_bound = 100
    lower_bound = upper_bound * (1 + detection_threshold)
    edges = cv2.Canny(contrast, upper_bound, lower_bound)

    # 3. Close gaps in edges (important!)
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Find contours
    contours, _ = cv2.findContours(
        edges_closed,
        cv2.RETR_EXTERNAL,  # only outer contours
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 5. Filter contours by size (remove small internal junk)
    min_area = 0.01 * image_matrix.shape[0] * \
        image_matrix.shape[1]  # 1% of image area
    outer_contours = [
        c for c in contours if cv2.contourArea(c) > min_area
    ]
    
    # task 1 : make outer contorus of type list that returns all outer non concentric contours
    processed_contours = []
    for cnt in outer_contours:
        pts = cnt.reshape(-1, 2).astype(float)
        pts = smooth_contours(pts)
        processed_contours.append(pts)
    
    # task 2 : iinstead of just one formatted_contour, draw all the outer non concentric contours
    processed_image = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
    for pts in processed_contours:
        formatted_contour = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.drawContours(processed_image, [formatted_contour], -1, (0, 255, 0), 2) 

    final_contours = []
    for pts in processed_contours:
        norm_pts = pts.copy()
        norm_pts[:, 0] = -1 + 2*norm_pts[:, 0]/(image_matrix.shape[1])
        norm_pts[:, 1] = -1 + 2*norm_pts[:, 1]/(image_matrix.shape[0])
        # norm_pts = norm_pts - np.mean(norm_pts, axis=0)
        norm_pts[:, 1] = -norm_pts[:, 1]
        final_contours.append(norm_pts)
    
    return final_contours, processed_image


def view_cell_image(processed_image, title = "viewed image"):
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


class Filter():
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.size = matrix.shape[0]

    def convolved(self, objective_matrix: np.ndarray, detection_threshold=1.5):
        m = self.matrix.shape[0]
        if np.sum(self.matrix * objective_matrix) >= detection_threshold*m:
            return True
        return False


def _line_filter(size=10):
    filter = np.zeros((size, size))
    for i in range(0, size):
        filter[i, i] = 1
        filter[i, size - i - 1] = 1
        filter[int(size/2), i] = 1
        filter[i, int(size/2)] = 1
    return filter


def detect_shapes_canon(image_matrix: np.ndarray, filter: Filter = Filter(_line_filter()), mode: str = "light", detection_threshold=0.2):
    if mode == "light":
        image_matrix = 1 - image_matrix/255
    image_dim = image_matrix.shape
    shape_points = []
    m = filter.size
    for i in range(0, int(image_dim[0]/m)-1):
        for j in range(0, int(image_dim[1]/m)-1):
            if filter.convolved(image_matrix[m*i: m*i+m, m*j: m*j+m], detection_threshold):
                image_matrix[m*i: m*i+m, m*j: m*j+m] = 0.5
                x = -1 + 2*j/(int(image_dim[1]/m)-1)
                y = 1 - 2*i/(int(image_dim[0]/m)-1)
                shape_points.append((x, y))
    shape_points = shape_points - np.mean(shape_points, axis=0)
    center = np.mean(shape_points, axis=0)
    Dx = (shape_points - center)[:, 0]
    Dy = (shape_points - center)[:, 1]
    angles = np.arctan2(Dy, Dx)
    n = shape_points.shape[0]
    stacked = np.zeros((n, 3))
    stacked[:, 0:2] = shape_points
    stacked[:, 2] = angles
    sorted_indices = np.argsort(stacked[:, 2])
    stacked = stacked[sorted_indices]
    shape_points = smooth_contours(stacked[:, 0:2])
    return shape_points, image_matrix
