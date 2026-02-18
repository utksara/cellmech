import numpy as np
import matplotlib.pyplot as plt
from cellmech.utils import symmetric_gaussian


def _getG(v: np.ndarray[tuple[int]], params: dict):
    x = v[0]
    y = v[1]
    p = params["p"]
    pi = params["pi"]
    E = params["E"]
    r = np.sqrt(x**2 + y**2) + 1e-4
    return (1 + p)/(pi * E * r**3) * np.array([
        [(1 - p) * r**2 + p * x**2, p*x*y],
        [p*x*y,                     (1 - p) * r**2 + p * y**2]
    ])


def _getG_Fourier(v: np.ndarray[tuple[int]], params: dict):
    x = v[0]
    y = v[1]
    p = params["p"]
    pi = params["pi"]
    E = params["E"]
    r = np.sqrt(x**2 + y**2) + 1e-4
    return 2*(1 + p)/(pi * E * r**3) * np.array([
        [(1 - p) * r**2 + p * y**2, -p*x*y],
        [-p*x*y,                    (1 - p) * r**2 + p * x**2]
    ])


def _matrix_2x2_inverse(A: np.ndarray[tuple[tuple[int]]]):
    factor = 1/(A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0])
    M = np.array([[A[1, 1], -A[0, 1]],
                  [-A[1, 0], A[0, 0]]])
    return factor*M


def _getGInv_Fourier(v, params):
    x = v[0]
    y = v[1]
    p = params["p"]
    pi = params["pi"]
    E = params["E"]
    r = np.sqrt(x**2 + y**2) + 1e-4
    M = np.array([
        [(1 - p) * r**2 + p * y**2, -p*x*y],
        [-p*x*y,  (1 - p) * r**2 + p * x**2]
    ])
    return (E * r**3)/(2*(1 + p)) * _matrix_2x2_inverse(M)


def _tensor_contraction(G_sub_matrix, force_field):
    S = np.einsum("lmij,lmj->i", G_sub_matrix, force_field)
    return S


class CellMechParameters():
    required_params = ["pi", "E", "pi", "N", "width", "pixel_size"]

    def __init__(self, params: dict):
        for key in self.required_params:
            if key not in params.keys():
                raise ValueError(key + " paramter missing!")
        self.params = params

def calculate_analytical_displacement(force_field,
                                      force_position,
                                      cellmechparams):
    """
    Analytical surface displacement from a point traction force
    using the 2D Boussinesq solution (half-space).

    Parameters
    ----------
    force_field : (N, N, 2) array
        Traction field (N/m^2 or N if single force).
    force_position : tuple (i0, j0)
        Pixel index where the point force is applied.
    cellmechparams : object
        Must contain:
            params["N"]  : grid size
            params["E"]  : Young's modulus (Pa)
            params["p"]  : Poisson ratio
            params["dx"] : physical pixel size (meters)
    """

    params = cellmechparams.params
    N = params["N"]
    E = params["E"]
    nu = params["p"]
    dx = params["width"]/N   # physical spacing in meters

    u = np.zeros((N, N, 2))

    # Extract applied force vector
    F = force_field[force_position[0], force_position[1]]
    Fx, Fy = F
    F_mag = np.sqrt(Fx**2 + Fy**2)
    
    # Physical position of applied force
    x0 = force_position[0] * dx
    y0 = force_position[1] * dx
    print("Analytical coefficient : ", F_mag * (1 + nu) / (2 * np.pi * E * dx))
    for i in range(N):
        for j in range(N):

            # Physical coordinates of evaluation point
            x = i * dx
            y = j * dx

            rx = x - x0
            ry = y - y0

            r2 = rx**2 + ry**2 + dx**2   # avoid singularity
            r = np.sqrt(r2)
            
            # Unit vectors
            r_unit = np.array([rx, ry])/r
            F_unit = np.array([rx, ry])/F_mag
                    
            # Boussinesq prefactor
            coeff = F_mag * (1 + nu) / (2 * np.pi * E * r)
            
            u[i, j, :] = coeff * ((3 - 4*nu) * F_unit + np.inner(F_unit, F_mag) * r_unit)

    return u


def calculate_dummy_force(force_points: list[tuple[float, float]], cellmechparams: CellMechParameters, custom_force: callable = symmetric_gaussian):
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
                dU[i, j, :] = custom_force(v1 - v2) * dx * dx
        U += dU
    return U


def fttc_force_to_displacement(F_field, E, nu, dx, pad_factor=2):
    N = F_field.shape[0]
    Npad = pad_factor * N

    # 1. Padding (Use 'start' and 'end' as you did, it helps with boundary artifacts)
    Fx_pad = np.zeros((Npad, Npad))
    Fy_pad = np.zeros((Npad, Npad))
    start = Npad // 2 - N // 2
    end = start + N
    Fx_pad[start:end, start:end] = F_field[:, :, 0]
    Fy_pad[start:end, start:end] = F_field[:, :, 1]

    # 2. Wavevectors - Crucial to use fftshift for centered padding
    freq = np.fft.fftfreq(Npad, d=dx)
    kx, ky = np.meshgrid(freq, freq)
    k2 = kx**2 + ky**2
    k = np.sqrt(k2)

    # 3. Green's Function Components
    # We use a tiny offset for k=0 to avoid NaN, then zero it out later
    k_safe = np.where(k == 0, 1e-10, k)
    
    # This is the standard Boussinesq solution for a half-space surface
    coeff = (2 * (1 + nu)) / (E * k_safe)
    
    G11 = coeff * (1 - nu + nu * (ky**2 / k_safe**2))
    G22 = coeff * (1 - nu + nu * (kx**2 / k_safe**2))
    G12 = coeff * (-nu * (kx * ky / k_safe**2))
    
    # Zero out the DC component (prevents infinite displacement/white screen)
    G11[0,0] = 0; G22[0,0] = 0; G12[0,0] = 0

    # 4. Transform Forces
    # Use ifftshift to move the centered padding to the corners for the FFT
    FT_Fx = np.fft.fft2(np.fft.ifftshift(Fx_pad))
    FT_Fy = np.fft.fft2(np.fft.ifftshift(Fy_pad))

    # 5. Multiplication
    # Force is in Newtons. To get displacement, we treat it as traction 
    # by dividing by the pixel area dx^2.
    FT_Ux = (G11 * FT_Fx + G12 * FT_Fy) / (dx**2)
    FT_Uy = (G12 * FT_Fx + G22 * FT_Fy) / (dx**2)

    # 6. Back to Real Space
    # Apply fftshift back to return the "center" to the center
    Ux_pad = np.real(np.fft.fftshift(np.fft.ifft2(FT_Ux)))
    Uy_pad = np.real(np.fft.fftshift(np.fft.ifft2(FT_Uy)))

    # 7. Crop and Return
    Ux = Ux_pad[start:end, start:end]
    Uy = Uy_pad[start:end, start:end]

    return np.stack([Ux, Uy], axis=-1)

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
    coeff = youngs_modulus * k_mag * 0.5 / (1 - poisson_ratio**2) 
    
    G11 = coeff * (1 - poisson_ratio + poisson_ratio * (ky**2 / k_mag**2))
    G12 = coeff * (poisson_ratio * (kx * ky / k_mag**2))
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


def calcualte_displacement(force_field: np.ndarray, cellmechparams: CellMechParameters, method: str = "fttc"):
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
                displacement[l, m, :] = _tensor_contraction(
                    G, force_field) * dx * dx
        return displacement

    if method == "fttc":
        return fttc_force_to_displacement(force_field, params["E"], params["p"], params["width"]/params["N"])


def plot_vector_field(force_field: np.ndarray, title=None):
    # params = cellmechparams.params
    # dim = params["width"]/2
    dim = 0.5
    N = force_field.shape[0]
    x = np.linspace(-dim, dim, N)
    y = np.linspace(-dim, dim, N)
    X, Y = np.meshgrid(x, y)
    plt.quiver(X, Y, force_field[:, :, 0], force_field[:, :, 1])
    if title is not None:
        plt.title(title)
    plt.show()
    

def calculate_constrained_traction_force(displacement: np.ndarray, cellmechparams: CellMechParameters):
    traction_force = calculate_traction_force(displacement, cellmechparams)
    for i in range(0, 2):
        recalc_displcemement = calcualte_displacement(traction_force, cellmechparams)
        recalc_traction_force = calculate_traction_force(recalc_displcemement, cellmechparams)
        mae = np.mean(np.abs(traction_force - recalc_traction_force))/np.mean(abs(traction_force))
        mme = np.mean(np.abs(traction_force - recalc_traction_force))/np.max(abs(traction_force))
        print(f'mean avg error {mae}, max error {mme} in iter {i}')
        traction_force = recalc_traction_force
    return traction_force

def calculate_traction_force(displacement: np.ndarray, cellmechparams: CellMechParameters):
    params = cellmechparams.params
    return fttc_displacement_to_force(displacement, params["pixel_size"], params["E"], params["p"])


def calculate_traction_force_tensor(displacement: np.ndarray, cellmechparams: CellMechParameters):
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
            D[l, m, :] = _tensor_contraction(G, U) * dx * dy

    # D = np.fft.ifft(D)
    # D = np.real(D)
    return D
