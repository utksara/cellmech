import numpy as np
import matplotlib.pyplot as plt
from cellmech.utils import symmetric_gaussian
        
def _getG(v : np.ndarray[tuple[int]], params : dict):
    x = v[0]
    y = v[1]
    p = params["p"]
    pi = params["pi"]
    E = params["E"]
    r = np.sqrt(x**2 + y**2) + 1e-4
    return (1 +  p)/(pi * E * r**3) * np.array([
        [(1 - p) * r**2 + p * x**2, p*x*y],
        [p*x*y,  (1 - p) * r**2 + p * y**2]
    ])
        
def _getG_Fourier(v : np.ndarray[tuple[int]], params : dict):
    x = v[0]
    y = v[1]
    p = params["p"]
    pi = params["pi"]
    E = params["E"]
    r = np.sqrt(x**2 + y**2) + 1e-4
    return 2*(1 +  p)/( E * r**3) * np.array([
        [(1 - p) * r**2 + p * y**2, -p*x*y],
        [-p*x*y,  (1 - p) * r**2 + p * x**2]
    ])
    
def _matrix_2x2_inverse(A : np.ndarray[tuple[tuple[int]]]):
    factor = 1/(A[0,0]*A[1,1] - A[0,1]*A[1,0])
    M = np.array([[A[1,1], -A[0,1]],
                  [-A[1,0], A[0,0]]])
    return factor*M

def _getGInv_Fourier(v, params):
    x = v[0]
    y = v[1]
    p = params["p"]
    pi = params["pi"]
    E = params["E"]
    r = np.sqrt(x**2 + y**2) + 1e-4
    M =  np.array([
        [(1 - p) * r**2 + p * y**2, -p*x*y],
        [-p*x*y,  (1 - p) * r**2 + p * x**2]
    ])
    return (E * r**3)/(2*(1 + p)) * _matrix_2x2_inverse(M)

def _tensor_contraction(G_sub_matrix, force_field):
    S = np.einsum("lmij,lmi->j", G_sub_matrix, force_field)
    return S

class CellMechParameters():
    required_params = ["pi", "E", "pi", "N", "width"]
    def __init__(self, params : dict):
        for key in self.required_params:
            if key not in params.keys():
                raise ValueError(key + " paramter missing!")
        self.params = params
        
class Filter():
    def __init__(self, matrix : np.ndarray):
        self.matrix = matrix
        self.size = matrix.shape[0]
        
    def convolved(self, objective_matrix : np.ndarray, detection_threshold = 1.5):
        m = self.matrix.shape[0]
        if np.sum(self.matrix * objective_matrix) >= detection_threshold*m:
            return True
        return False
    
def calculate_dummy_force(force_points : list[tuple[float, float]], cellmechparams: CellMechParameters, custom_force : callable = symmetric_gaussian):
    params = cellmechparams.params
    N = params["N"]
    dx = params["width"]/N
    dim = params["width"]/2
    X = np.linspace(-dim, dim, N)
    U = np.zeros((N, N, 2))
    for point in force_points:
        dU = np.zeros((N, N, 2))
        v2 = np.array(point)
        for i in range(0, N):
            for j in range(0, N):
                v1 = np.array((X[i], X[j]))
                dU[i, j, :] = custom_force(v1 - v2)  * dx * dx
        U += dU
    return U

def calcualte_displacement(force_field : np.ndarray, cellmechparams: CellMechParameters, method : str = "fttc"):  
    params = cellmechparams.params
    if method == "tn": 
        N = force_field.shape[0]
        dim = params["width"]/2
        dx = params["width"]/N 
        G_matrix = np.zeros((2*N - 1, 2*N - 1, 2, 2))
        displacement = np.zeros((N, N, 2))
        force_field = np.fft.fft(force_field)
        X_ext = np.linspace(-2*dim + dim/N, 2*dim - dim/N, 2*N - 1)
        for l in range(0, 2*N - 1):
            for m in range(0, 2*N - 1):
                v = np.array((X_ext[l], X_ext[m]))
                G_matrix[l, m, :, :] = _getG(v, params)
            
        for l in range(0, N - 1):
            for m in range(0, N - 1):
                v = np.array((X_ext[l], X_ext[m]))
                G = G_matrix[l:l + N, m:m + N, :, :]
                displacement[l, m, :] = _tensor_contraction(G, force_field)  * dx * dx
        return displacement
    
    if method == "fttc":
        N = force_field.shape[0]
        dim = params["width"]/2
        displacement = np.zeros((N, N, 2))
        force_field = np.real(np.fft.fft(force_field))
        X_ext = np.linspace(-2*dim + dim/N, 2*dim - dim/N, 2*N - 1)
        for l in range(0, N - 1):
            for m in range(0, N - 1):
                v = np.array((X_ext[l], X_ext[m]))
                displacement[l, m, :] = _getG_Fourier(v, params) @ force_field[l, m, :]
        displacement = np.real(np.fft.ifft(displacement))
        return displacement

def plot_vector_field(force_field : np.ndarray, cellmechparams: CellMechParameters, title = None):
    params = cellmechparams.params
    dim = params["width"]/2
    N = force_field.shape[0]
    x = np.linspace(-dim, dim, N)
    y = np.linspace(-dim ,dim , N)
    X, Y = np.meshgrid(x, y)
    plt.quiver(X, Y, force_field[:, :, 0], force_field[:, :, 1])
    if title is not None:
        plt.title(title)
    plt.show()


def _line_filter(size = 10):
    filter = np.zeros((size, size))
    for i in range(0, size):
        filter[i, i] = 1
        filter[i, size - i - 1] = 1
        filter[int(size/2), i] = 1
        filter[i, int(size/2)] = 1
    return filter

def detect_shapes(image_matrix : np.ndarray, filter : Filter = Filter(_line_filter()), mode : str = "light", detection_threshold = 0.2):
    if mode == "light":
        image_matrix = 1 - image_matrix/255
    image_dim = image_matrix.shape
    shape_points = []
    m = filter.size
    for i in range(0, int(image_dim[0]/m)-1):
        for j in range(0, int(image_dim[1]/m)-1):
            if filter.convolved(image_matrix[m*i: m*i+m, m*j: m*j+m], detection_threshold):
                image_matrix[m*i: m*i+m, m*j: m*j+m] = 0.5
                x =  -1 + 2*i/(int(image_dim[0]/m)-1)
                y =  -1 + 2*j/(int(image_dim[1]/m)-1)
                shape_points.append((x, y))
    return shape_points, image_matrix
    
def calculate_traction_force(displacement : np.ndarray, cellmechparams: CellMechParameters):
    params = cellmechparams.params
    Nx = displacement.shape[0]
    Ny = displacement.shape[1] 
    U = np.real(np.fft.fft(displacement))
    G_matrix = np.zeros((2*Nx - 1, 2*Ny - 1, 2, 2))
    D = np.zeros((Nx, Ny, 2))
    dim = params["width"]/2
    dx = params["width"]/Nx
    dy = params["width"]/Ny
    
    X_ext = np.linspace(-2*dim + dim/Nx, 2*dim - dim/Nx, 2*Nx - 1)
    # for l in range(0, 2*Nx - 1):
    #     for m in range(0, 2*Ny - 1):
    #         v = np.array((X_ext[l], X_ext[m]))
    #         G_matrix[l, m, :, :] = _getGInv_Fourier(v, params)
        
    for l in range(0, Nx - 1):
        for m in range(0, Ny - 1):
            # G = G_matrix[l:l + Nx, m:m + Ny, :, :]
            v = np.array((X_ext[l], X_ext[m]))
            D[l, m, :] = _getGInv_Fourier(v,params) @ U[l, m, :]
            # D[l, m, :] = _tensor_contraction(G, U)  * dx * dy
    
    D = np.fft.ifft(D)
    D = np.real(D)
    return D

