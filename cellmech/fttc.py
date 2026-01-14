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
        [(1 - p) * r**2 + p * x**2, p*x*y                    ],
        [p*x*y,                     (1 - p) * r**2 + p * y**2]
    ])
        
def _getG_Fourier(v : np.ndarray[tuple[int]], params : dict):
    x = v[0]
    y = v[1]
    p = params["p"]
    pi = params["pi"]
    E = params["E"]
    r = np.sqrt(x**2 + y**2) + 1e-4
    return 2*(1 +  p)/(pi * E * r**3) * np.array([
        [(1 - p) * r**2 + p * y**2, -p*x*y                   ],
        [-p*x*y,                    (1 - p) * r**2 + p * x**2]
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
    required_params = ["pi", "E", "pi", "N", "width", "pixel_size"]
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


def fttc_displacement_to_force(U_field, pixel_size, youngs_modulus, poisson_ratio, reg_param=1e-10):
    """
    Converts a 2D displacement field to a traction force field using 
    regularized Fourier Transform Traction Cytometry (FTTC).
    
    Parameters:
    U_field: ndarray of shape (N, N, 2) -> [Ux, Uy] in meters
    pixel_size: meters per pixel
    youngs_modulus: E in Pascals
    poisson_ratio: nu (typically 0.5)
    reg_param: Lambda (regularization parameter)
    """
    N = U_field.shape[0]
    Ux = U_field[:, :, 0]
    Uy = U_field[:, :, 1]

    # 1. Setup Frequency Space
    freq = np.fft.fftfreq(N, d=pixel_size)
    kx, ky = np.meshgrid(freq, freq)
    k_mag = np.sqrt(kx**2 + ky**2)
    k_mag[0, 0] = np.inf # Avoid division by zero

    # 2. Fourier Transform of Displacements
    FT_Ux = np.fft.fft2(Ux)
    FT_Uy = np.fft.fft2(Uy)

    # 3. Green's Tensor in Fourier Space (Boussinesq)
    # Pre-factor: G_ij = coeff * M_ij
    coeff = (1 + poisson_ratio) / (np.pi * youngs_modulus * k_mag)
    
    G11 = coeff * (1 - poisson_ratio + poisson_ratio * (ky**2 / k_mag**2))
    G12 = coeff * (-poisson_ratio * (kx * ky / k_mag**2))
    G21 = G12
    G22 = coeff * (1 - poisson_ratio + poisson_ratio * (kx**2 / k_mag**2))

    # 4. Tikhonov Regularized Inversion
    # We solve: F = (G.T * G + lambda^2 * I)^-1 * G.T * U
    # Since G is a 2x2 matrix at each k-point, we can solve it analytically.
    
    # Determinant of (G^T G + lambda^2 I)
    # Note: For symmetric G, G^T G is just G squared.
    det = (G11**2 + G12**2 + reg_param**2) * (G22**2 + G21**2 + reg_param**2) - \
          (G11*G21 + G12*G22)**2

    # Applying the inverse filter
    FT_Fx = ((G11 * FT_Ux + G21 * FT_Uy) * (G22**2 + G21**2 + reg_param**2) - 
             (G12 * FT_Ux + G22 * FT_Uy) * (G11*G21 + G12*G22)) / det
             
    FT_Fy = ((G12 * FT_Ux + G22 * FT_Uy) * (G11**2 + G12**2 + reg_param**2) - 
             (G11 * FT_Ux + G21 * FT_Uy) * (G11*G21 + G12*G22)) / det

    # Handle the DC component
    FT_Fx[0, 0] = 0
    FT_Fy[0, 0] = 0

    # 5. Inverse FFT to Real Space
    Fx = np.real(np.fft.ifft2(FT_Fx))
    Fy = np.real(np.fft.ifft2(FT_Fy))

    return np.stack([Fx, Fy], axis=-1)

def fttc_force_to_displacement(F_field, pixel_size, youngs_modulus, poisson_ratio):
    """
    Converts a 2D force field to a displacement field using the 
    Boussinesq Green's function in the Fourier domain.
    
    Parameters:
    F_field: ndarray of shape (N, N, 2) -> [Fx, Fy]
    pixel_size: physical size of one pixel (e.g., meters)
    youngs_modulus: E in Pascals
    poisson_ratio: nu (typically 0.5 for hydrogels)
    """
    N = F_field.shape[0]
    Fx = F_field[:, :, 0]
    Fy = F_field[:, :, 1]

    # 1. Create frequency coordinates
    freq = np.fft.fftfreq(N, d=pixel_size)
    kx, ky = np.meshgrid(freq, freq)
    
    # Calculate k magnitude (add epsilon to avoid division by zero at origin)
    k_mag = np.sqrt(kx**2 + ky**2)
    k_mag[0, 0] = np.inf 

    # 2. Fourier Transform of the Force components
    FT_Fx = np.fft.fft2(Fx)
    FT_Fy = np.fft.fft2(Fy)

    # 3. Define the Green's Function components in Fourier Space
    # Pre-factor for Boussinesq solution
    coeff = (1 + poisson_ratio) / (np.pi * youngs_modulus * k_mag)
    
    # Green's tensor components
    G11 = coeff * (1 - poisson_ratio + poisson_ratio * (ky**2 / k_mag**2))
    G12 = coeff * (-poisson_ratio * (kx * ky / k_mag**2))
    G21 = G12
    G22 = coeff * (1 - poisson_ratio + poisson_ratio * (kx**2 / k_mag**2))

    # 4. Calculate Displacement in Fourier Domain: U = G * F
    FT_Ux = G11 * FT_Fx + G12 * FT_Fy
    FT_Uy = G21 * FT_Fx + G22 * FT_Fy

    # Handle the DC component (zero frequency)
    FT_Ux[0, 0] = 0
    FT_Uy[0, 0] = 0

    # 5. Inverse Fourier Transform to get real-space displacement
    Ux = np.real(np.fft.ifft2(FT_Ux))
    Uy = np.real(np.fft.ifft2(FT_Uy))

    return np.stack([Ux, Uy], axis=-1)

def calcualte_displacement(force_field : np.ndarray, cellmechparams: CellMechParameters, method : str = "fttc"):  
    params = cellmechparams.params
    if method == "tn": 
        N = force_field.shape[0]
        dim = params["width"]/2
        dx = params["width"]/N 
        G_matrix = np.zeros((2*N - 1, 2*N - 1, 2, 2))
        displacement = np.zeros((N, N, 2))
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
        return fttc_force_to_displacement(force_field, params["pixel_size"], params["E"], params["p"])

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
    return fttc_displacement_to_force(displacement, params["pixel_size"], params["E"], params["p"])

def calculate_traction_force_tensor(displacement : np.ndarray, cellmechparams: CellMechParameters):
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
    for l in range(0, 2*Nx - 1):
        for m in range(0, 2*Ny - 1):
            v = np.array((X_ext[l], X_ext[m]))
            G_matrix[l, m, :, :] = _matrix_2x2_inverse(_getG(v, params))
        
    for l in range(0, Nx - 1):
        for m in range(0, Ny - 1):
            G = G_matrix[l:l + Nx, m:m + Ny, :, :]
            v = np.array((X_ext[l], X_ext[m]))
            # D[l, m, :] = _getGInv_Fourier(v,params) @ U[l, m, :]
            D[l, m, :] = _tensor_contraction(G, U)  * dx * dy
    
    # D = np.fft.ifft(D)
    # D = np.real(D)
    return D

