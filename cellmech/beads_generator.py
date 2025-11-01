import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_point_position(bead_matrix):
    m, n = bead_matrix.shape
    point_position = []
    for i in range(0, m):
        for j in range(0, n):
            if bead_matrix[i, j] == 1:
                point_position.append(np.array([i, j]))
    return point_position

def unroll(bead_matrix, assumed_center):
    point_positions = get_point_position(bead_matrix)
    unrolled_matrix = np.zeros((len(point_positions), 2))
    print(point_positions)
    for i in range(0, len(point_positions)):
        point = point_positions[i]
        distance = np.linalg.norm(point - assumed_center)
        angle = np.arccos((point[1] - assumed_center[1])/distance)
        print(angle * 180/np.pi, "deg", point)
        if (point[0] - assumed_center[0] < 0):
            angle = angle + np.pi
        unrolled_matrix[i, 0] = angle
        unrolled_matrix[i, 1] = distance
    return unrolled_matrix

def create_random_bead(grid : int, interval : int):
    if (interval > grid/2):
        ValueError("Invalid interval size, must be smaller than grid/2")
    
    bead_matrix = np.zeros((grid, grid))
    sub_grid = int(grid/interval)
    for i in range(0, interval - 1):
        for j in range(0, interval - 1):
            m = np.random.randint(0, sub_grid - 1)
            n = np.random.randint(0, sub_grid - 1)
            bead_matrix[i*sub_grid + m, j*sub_grid + n] = 1
            
    return bead_matrix




        