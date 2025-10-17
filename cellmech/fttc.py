import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
        
def cellmech_paramaters(params):
    for key in ["pi", "E", "pi", "N", "width"]:
        if key not in params.keys():
            raise ValueError(key + " paramter missing!")
    return params
    
def getG(v : np.ndarray[tuple[int]], params):
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
        
def getG_Fourier(v : np.ndarray[tuple[int]], params):
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
    
def matrix_inverse(A):
    factor = 1/(A[0,0]*A[1,1] - A[0,1]*A[1,0])
    M = np.array([[A[1,1], -A[0,1]],
                  [-A[1,0], A[0,0]]])
    return factor*M

def getGInv_Fourier(v, params):
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
    return (E * r**3)/(2*(1 + p)) * matrix_inverse(M)

def symmetric_gaussian(v):
    r2 = v[0]**2 + v[1]**2
    r = np.sqrt(r2)
    A = 10e-4 * np.exp(-10*r2)
    cos = -v[0]/r
    sin = -v[1]/r
    return np.array([A*cos, A*sin])

def get_force_field(force_points, params, custom_force = symmetric_gaussian):
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

def tensor_contraction(G_sub_matrix, force_field):
    S = np.einsum("lmij,lmi->j", G_sub_matrix, force_field)
    return S

def calcualte_displacement(force_field, params):   
    N = force_field.shape[0]
    dim = params["width"]/2
    dx = params["width"]/N 
    G_matrix = np.zeros((2*N - 1, 2*N - 1, 2, 2))
    D = np.zeros((N, N, 2))
    force_field = np.fft.fft(force_field)
    X_ext = np.linspace(-2*dim + dim/N, 2*dim - dim/N, 2*N - 1)
    for l in range(0, 2*N - 1):
        for m in range(0, 2*N - 1):
            v = np.array((X_ext[l], X_ext[m]))
            G_matrix[l, m, :, :] = getG(v, params)
        
    for l in range(0, N - 1):
        for m in range(0, N - 1):
            v = np.array((X_ext[l], X_ext[m]))
            G = G_matrix[l:l + N, m:m + N, :, :]
            D[l, m, :] = tensor_contraction(G, force_field)  * dx * dx
          
    return D

def calcualte_displacement_efficient(force_field, params):   
    N = force_field.shape[0]
    dim = params["width"]/2
    D = np.zeros((N, N, 2))
    force_field = np.fft.fft(force_field)
    X_ext = np.linspace(-2*dim + dim/N, 2*dim - dim/N, 2*N - 1)
    for l in range(0, N - 1):
        for m in range(0, N - 1):
            v = np.array((X_ext[l], X_ext[m]))
            D[l, m, :] = getG_Fourier(v, params) @ force_field[l, m, :]
    D = np.fft.ifft(D)
    D = np.real(D)            
    return D

def plot_field(force_field, params, title = None):
    dim = params["width"]/2
    N = force_field.shape[0]
    x = np.linspace(-dim, dim, N)
    y = np.linspace(-dim ,dim , N)
    X, Y = np.meshgrid(x, y)
    plt.quiver(X, Y, force_field[:, :, 0], force_field[:, :, 1])
    if title is not None:
        plt.title(title)
    plt.show()

def line_filter(size = 10):
    filter = np.zeros((size, size))
    for i in range(0, size):
        filter[i, i] = 1
        filter[i, size - i - 1] = 1
        filter[int(size/2), i] = 1
        filter[i, int(size/2)] = 1
    return filter

def detect_shapes(image_matrix : np.array, filter : np.array, mode : str = "light"):
    if mode == "light":
        image_matrix = 1 - image_matrix/255
    image_dim = image_matrix.shape
    shape_points = []
    m = filter.shape[0]
    for i in range(0, int(image_dim[0]/m)-1):
        for j in range(0, int(image_dim[1]/m)-1):
            if np.sum(filter * image_matrix[m*i: m*i+m, m*j: m*j+m]) >= 1.5* m:
                image_matrix[m*i: m*i+m, m*j: m*j+m] = 0.5
                x =  -1 + 2*i/(int(image_dim[0]/m)-1)
                y =  -1 + 2*j/(int(image_dim[1]/m)-1)
                shape_points.append((x, y))
    return shape_points, image_matrix
    
def calcualte_traction(disp_field, params):
    Nx = disp_field.shape[0]
    Ny = disp_field.shape[1] 
    U = np.real(np.fft.fft(disp_field))
    G_matrix = np.zeros((2*Nx - 1, 2*Ny - 1, 2, 2))
    D = np.zeros((Nx, Ny, 2))
    dim = params["width"]/2
    dx = params["width"]/Nx
    dy = params["width"]/Ny
    
    X_ext = np.linspace(-2*dim + dim/Nx, 2*dim - dim/Nx, 2*Nx - 1)
    # for l in range(0, 2*Nx - 1):
    #     for m in range(0, 2*Ny - 1):
    #         v = np.array((X_ext[l], X_ext[m]))
    #         G_matrix[l, m, :, :] = getGInv_Fourier(v, params)
        
    for l in range(0, Nx - 1):
        for m in range(0, Ny - 1):
            # G = G_matrix[l:l + Nx, m:m + Ny, :, :]
            v = np.array((X_ext[l], X_ext[m]))
            D[l, m, :] = getGInv_Fourier(v,params) @ U[l, m, :]
            # D[l, m, :] = tensor_contraction(G, U)  * dx * dy
    
    D = np.fft.ifft(D)
    D = np.real(D)
    return D

