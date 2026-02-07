import numpy as np
import cv2
import matplotlib.pyplot as plt


def smooth_contours(outer_contours):
    smoothed_contours = []

    for c in outer_contours:
        epsilon = 0.01 * cv2.arcLength(c, True)  # 1% of perimeter
        c_smooth = cv2.approxPolyDP(c, epsilon, True)
        smoothed_contours.append(c_smooth)
    return smoothed_contours


def detect_shapes(image_matrix: np.ndarray, detection_threshold=0.5):

    # # Load image (grayscale)
    # img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # if img is None:
    #     raise FileNotFoundError("Could not load image")

    # 1. Improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(image_matrix)

    # 2. Edge detection

    upper_bound = 150
    lower_bound = 150 * (1 + detection_threshold)
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

    # 6. Draw only outer contours
    processed_image = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
    # outer_contours = smooth_contours(outer_contours)
    cv2.drawContours(processed_image, outer_contours, -1, (0, 255, 0), 2)

    outer_contours = np.array(outer_contours[0])
    m = outer_contours.shape[0]
    outer_contours = outer_contours.reshape(m, 2)
    
    outer_contours = outer_contours - np.mean(outer_contours, axis = 0)
    outer_contours[:,0] = 0.5*outer_contours[:,0]/(np.max(outer_contours[:,0]))
    outer_contours[:,1] = 0.5*outer_contours[:,1]/(np.max(outer_contours[:,1]))
    return outer_contours, processed_image


def plot_results(processed_image):
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap="gray")
    plt.title("Detected cell Boundary")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# for filename in os.listdir("./cell_images/"):
#     print(filename)
#     filter_image(f'./cell_images/{filename}')

# filter_image(f'./cell_images/imgline.png')

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


def detect_shapes_deprecated(image_matrix: np.ndarray, filter: Filter = Filter(_line_filter()), mode: str = "light", detection_threshold=0.2):
    if mode == "light":
        image_matrix = 1 - image_matrix/255
    image_dim = image_matrix.shape
    shape_points = []
    m = filter.size
    for i in range(0, int(image_dim[0]/m)-1):
        for j in range(0, int(image_dim[1]/m)-1):
            if filter.convolved(image_matrix[m*i: m*i+m, m*j: m*j+m], detection_threshold):
                image_matrix[m*i: m*i+m, m*j: m*j+m] = 0.5
                x = -1 + 2*i/(int(image_dim[0]/m)-1)
                y = -1 + 2*j/(int(image_dim[1]/m)-1)
                shape_points.append((x, y))
    return np.array(shape_points), image_matrix
