import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from cellmech.utils import simple_unit_force


def getG(v: np.ndarray[tuple[int]], params: dict):
    x = v[0]
    y = v[1]
    p = params["p"]
    pi = params["pi"]
    E = params["E"]
    r = np.sqrt(x**2 + y**2)
    if r<=1e-3 : return np.zeros((2,2))
    return (1 + p)/(pi * E * r**3) * np.array([
        [(1 - p) * r**2 + p * x**2, p*x*y],
        [p*x*y, (1 - p) * r**2 + p * y**2]
    ])


def getG_Fourier(v: np.ndarray[tuple[int]], params: dict):
    x = v[0]
    y = v[1]
    p = params["p"]
    pi = params["pi"]
    E = params["E"]
    r = np.sqrt(x**2 + y**2)
    if r<=1e-3 : return np.zeros((2,2))
    return 2*(1 + p)/(pi * E * r**3) * np.array([
        [(1 - p) * r**2 + p * y**2, -p*x*y],
        [-p*x*y,                    (1 - p) * r**2 + p * x**2]
    ])


def _matrix_2x2_inverse(A: np.ndarray[tuple[tuple[int]]]):
    factor = 1/(A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0])
    M = np.array([[A[1, 1], -A[0, 1]],
                  [-A[1, 0], A[0, 0]]])
    return factor*M


def getGInv_Fourier(v, params):
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


def tensor_contraction(G_sub_matrix, force_field):
    S = np.einsum("lmij,lmi->j", G_sub_matrix, force_field)
    return S

class CellMechParameters():
    required_params = ["E", "pi", "N", "width", "pixel_size"]

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

# task :  modify  this function such that force_points can be a list of force_ponts representing each non concentric outer contour as the implementation in detect_cell
def calculate_dummy_force(force_points: list, cellmechparams: CellMechParameters, custom_force: callable = simple_unit_force):
    params = cellmechparams.params
    scaling = 1e-6 # force in micro newtons
    N = params["N"]
    dx = params["width"]/N
    dim = params["width"]/2
    U = np.zeros((N, N, 2))

    # Support for multiple contours
    if isinstance(force_points, list) and len(force_points) > 0 and isinstance(force_points[0], np.ndarray):
        contours = force_points
    else:
        contours = [np.array(force_points)]

    # task 1: in case the force_points is of type list of force_points, add another loop which iterate through 
    #         each outer contour in force_points
    
    force_location = np.zeros([N, N])
    for contour in contours:
        contour_arr = np.array(contour)
        v_centr = np.mean(contour_arr, axis=0)
        print(v_centr)
        ic = int((v_centr[0] + 1)*(N-1)/2)
        jc = int((v_centr[1] + 1)*(N-1)/2)
        force_location[ic, jc] = 1
        # Central forces (expansion/contraction) for this contour
        if 0 < ic < N-1 and 0 < jc < N-1:
            U[ic - 1, jc, :] += custom_force(np.array([-1, 0]))
            U[ic, jc + 1, :] += custom_force(np.array([0, 1]))
            U[ic, jc - 1, :] += custom_force(np.array([0, -1]))
            U[ic + 1, jc, :] += custom_force(np.array([1, 0]))

        for point in contour_arr:
            v2 = np.array(point)
            # finding the position of the force points
            # print(v2)
            i = int((v2[0] + 1)*(N-1)/2)
            j = int((v2[1] + 1)*(N-1)/2)
            force_location[i, j] = 1
            if 0 <= i < N and 0 <= j < N:
                U[i, j, :] = custom_force(v_centr - v2)
                
    return U * scaling, force_location


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



def calculate_tensile_force_magnitude(u_field, cellmechparams, sigma=0):
    """
    Computes the maximum principal strain (tensile strain) field.
    
    Parameters:
    -----------
    u_field : (N, N, 2) array
        Displacement field (Ux, Uy).
    dx : float
        Physical pixel size.
    sigma : float
        Standard deviation for Gaussian smoothing (0 = no smoothing).
        
    Returns:
    --------
    eps_1 : (N, N) array
        Maximum principal strain field.
    """
    params = cellmechparams.params
    ux = u_field[:, :, 0]
    uy = u_field[:, :, 1]
    N = u_field.shape[0]
    dx = params["width"]/N 
    # Optional smoothing to prevent gradient noise
    if sigma > 0:
        ux = gaussian_filter(ux, sigma=sigma)
        uy = gaussian_filter(uy, sigma=sigma)

    # 1. Compute Gradients
    # axis 0 is y (rows), axis 1 is x (cols)
    dux_dy, dux_dx = np.gradient(ux, dx)
    duy_dy, duy_dx = np.gradient(uy, dx)

    # 2. Define Strain Tensor Components
    exx = dux_dx
    eyy = duy_dy
    exy = 0.5 * (dux_dy + duy_dx)

    # 3. Calculate First Principal Strain (Maximum Tension)
    # This formula finds the eigenvalues of the 2D strain tensor
    mean_strain = (exx + eyy) / 2
    diff_strain = np.sqrt(((exx - eyy) / 2)**2 + exy**2)
    
    eps_1 = mean_strain + diff_strain
    
    return eps_1

def fttc_displacement_to_force(U_field, pixel_size, youngs_modulus, poisson_ratio, reg_param=1e-10):
    N = U_field.shape[0]
    Ux = U_field[:, :, 0]
    Uy = U_field[:, :, 1]

    # 1. Setup Frequency Space (Angular frequency)
    freq = np.fft.fftfreq(N, d=pixel_size)
    kx, ky = np.meshgrid(freq, freq)
    k_mag = np.sqrt(kx**2 + ky**2)
    
    # Avoid division by zero for the Green's function
    k_safe = np.where(k_mag == 0, 1e-10, k_mag)

    # 2. Fourier Transform of Displacements
    FT_Ux = np.fft.fft2(Ux)
    FT_Uy = np.fft.fft2(Uy)

    # 3. Define the Forward Green's Tensor components (Traction -> Displacement)
    # G_tilde = coeff * [[M11, M12], [M21, M22]]
    # Pre-factor for Boussinesq surface: 2(1+nu)/(E*k)
    coeff = (2 * (1 + poisson_ratio)) / (youngs_modulus * k_safe)
    
    G11 = coeff * (1 - poisson_ratio + poisson_ratio * (ky**2 / k_safe**2))
    G22 = coeff * (1 - poisson_ratio + poisson_ratio * (kx**2 / k_safe**2))
    G12 = coeff * (-poisson_ratio * (kx * ky / k_safe**2))
    G21 = G12

    # 4. Tikhonov Regularized Inversion
    # Formula: FT_F = inv(G^2 + lambda^2 * I) * G * FT_U
    # This effectively solves the inverse while suppressing high-frequency noise
    
    # Common denominator for the regularized inversion
    # We use G_mag_sq + reg^2 to dampen the inversion of small G values
    G_det = G11 * G22 - G12 * G21
    denom = G11**2 + G22**2 + 2*G12**2 + reg_param**2
    
    # FT_F = (G_transpose / (G^2 + reg^2)) * FT_U
    # For a more robust inversion, we calculate the inverse of G directly 
    # and apply a Tikhonov window:
    
    # Analytical 2x2 inverse components for G
    invG11 = G22 / (G_det + reg_param)
    invG22 = G11 / (G_det + reg_param)
    invG12 = -G12 / (G_det + reg_param)
    
    # Apply high-frequency rollout/regularization filter
    # This prevents the 1/k singularity from blowing up noise
    reg_filter = k_mag**2 / (k_mag**2 + reg_param**2)
    
    FT_Fx = (invG11 * FT_Ux + invG12 * FT_Uy) * reg_filter
    FT_Fy = (invG12 * FT_Ux + invG22 * FT_Uy) * reg_filter

    # Handle the DC component (no net force)
    FT_Fx[0, 0] = 0
    FT_Fy[0, 0] = 0

    # 5. Inverse FFT to Real Space
    # Note: No need to divide by pixel_size**2 here if Ux/Uy are in meters,
    # as the Green's function units (m/Pa) handle the conversion to Pascals.
    Fx = np.real(np.fft.ifft2(FT_Fx))
    Fy = np.real(np.fft.ifft2(FT_Fy))

    return np.stack([Fx, Fy], axis=-1)


def calculate_displacement(force_field: np.ndarray, cellmechparams: CellMechParameters, method: str = "fttc"):
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
                G_matrix[l, m, :, :] = getG(v, params)

        for l in range(0, N - 1):
            for m in range(0, N - 1):
                G = G_matrix[int((2*N-1)/2) - l: int((2*N-1)/2) - l + N, int((2*N-1)/2) - m: int((2*N-1)/2) - m + N, :, :]
                displacement[l, m, :] = tensor_contraction(G, force_field)  * dx * dx
        return displacement
    
    if method == "fttc":
        return fttc_force_to_displacement(force_field, params["E"], params["p"], params["width"]/params["N"])


def plot_vector_field(force_field: np.ndarray, title=None, custom_scale = None):
    dim = 0.5
    N = force_field.shape[0]
    x = np.linspace(-dim, dim, N)
    y = np.linspace(-dim, dim, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    force_magn = np.sqrt(force_field[:, :, 0]**2 + force_field[:,  :, 1]**2)
    max_force = np.max(force_magn)
    min_force = np.min(force_magn)
    if custom_scale is None:
        scaling_force = max_force
    else : scaling_force = custom_scale 

    plt.quiver(X, Y, force_field[:, :, 0]/scaling_force, 
               force_field[:, :, 1]/scaling_force, angles='xy', scale_units='xy', scale=N * max_force/scaling_force)
    
    full_title = f"Max: {max_force:.2e}, Min: {min_force:.2e}"
    if title is not None:
        full_title = f"{title}\n{full_title}"
    
    plt.title(full_title)
    plt.xlabel("X - axis")
    plt.ylabel("Y - axis")
    plt.show()
    

def calculate_constrained_traction_force(displacement: np.ndarray, cellmechparams: CellMechParameters):
    traction_force = calculate_traction_force(displacement, cellmechparams)
    for _ in range(0, 2):
        recalc_displcemement = calculate_displacement(traction_force, cellmechparams)
        traction_force = calculate_traction_force(recalc_displcemement, cellmechparams)
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
            G_matrix[l, m, :, :] = _matrix_2x2_inverse(getG(v, params))

    for l in range(0, Nx - 1):
        for m in range(0, Ny - 1):
            G = G_matrix[l:l + Nx, m:m + Ny, :, :]
            v = np.array((X_ext[l], X_ext[m]))
            # D[l, m, :] = getGInv_Fourier(v,params) @ U[l, m, :]
            D[l, m, :] = tensor_contraction(G, U) * dx * dy

    # D = np.fft.ifft(D)
    # D = np.real(D)
    return D

def calculate_traction_below(surface_traction : np.ndarray, depth :float, m : int, cellmechparams : CellMechParameters):
    E = cellmechparams.params["N"]
    nu = cellmechparams.params["p"]
    return calculate_traction_euler(depth, surface_traction, E, nu, m)

def calculate_traction_euler(depth, surface_traction, E, nu, m):
    """
    Compute shear stress (sigma_xz, sigma_yz) at different depths
    from surface traction using elastic half-space theory.

    Returns:
    T_depth: (N, N, m, 2)
    """
    
    N = surface_traction.shape[0]
    dx = 1.0  # assume unit spacing unless you pass it explicitly
    dz = depth / (m - 1)
    
    mu = E / (2 * (1 + nu))
    
    # Fourier frequencies
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1e-9  # avoid division by zero
    
    # FFT of surface traction
    Tx_hat = np.fft.fft2(surface_traction[:, :, 0])
    Ty_hat = np.fft.fft2(surface_traction[:, :, 1])
    
    # Surface displacement (approximate kernel)
    Ux0_hat = Tx_hat / (mu * K)
    Uy0_hat = Ty_hat / (mu * K)
    
    # Depth grid
    z_vals = np.linspace(0, depth, m)
    
    T_depth = np.zeros((N, N, m, 2))
    
    for i, z in enumerate(z_vals):
        decay = np.exp(-K * z)
        
        # displacement at depth
        Ux_hat = Ux0_hat * decay
        Uy_hat = Uy0_hat * decay
        
        # derivative w.r.t z in Fourier space: d/dz = -K
        dUx_dz_hat = -K * Ux_hat
        dUy_dz_hat = -K * Uy_hat
        
        # shear stress: sigma_xz = mu * dux/dz
        sigma_xz_hat = mu * dUx_dz_hat
        sigma_yz_hat = mu * dUy_dz_hat
        
        # back to real space
        T_depth[:, :, i, 0] = np.real(np.fft.ifft2(sigma_xz_hat))
        T_depth[:, :, i, 1] = np.real(np.fft.ifft2(sigma_yz_hat))
    
    return T_depth

def compute_stress_from_displacement(U, E, nu, dx, dz):
    """
    Compute stress tensor from displacement field U.

    U shape: (N, N, m, 2)  # (ux, uy)
    Returns: sigma_xz, sigma_yz (shear components vs depth)
    """
    
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    # Gradients
    dux_dx = np.gradient(U[:, :, :, 0], dx, axis=0)
    dux_dy = np.gradient(U[:, :, :, 0], dx, axis=1)
    dux_dz = np.gradient(U[:, :, :, 0], dz, axis=2)
    
    duy_dx = np.gradient(U[:, :, :, 1], dx, axis=0)
    duy_dy = np.gradient(U[:, :, :, 1], dx, axis=1)
    duy_dz = np.gradient(U[:, :, :, 1], dz, axis=2)
    
    # Shear stresses (what you probably want)
    sigma_xz = mu * dux_dz
    sigma_yz = mu * duy_dz
    
    return sigma_xz, sigma_yz

def calculate_displacement_depth(depth, surface_traction, E, nu, m, dx):
    """
    Compute displacement field inside elastic half-space from surface traction.

    Parameters:
    - depth: total depth
    - surface_traction: (N, N, 2)
    - E, nu: material properties
    - m: number of depth slices
    - dx: spatial resolution

    Returns:
    - U: (N, N, m, 2)
    """
    
    N = surface_traction.shape[0]
    mu = E / (2 * (1 + nu))
    
    # Fourier frequencies
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    K[0,0] = 1e-9  # avoid division by zero

    # FFT of traction
    T_hat_x = np.fft.fft2(surface_traction[:, :, 0])
    T_hat_y = np.fft.fft2(surface_traction[:, :, 1])

    # --- Surface displacement from traction (simplified kernel) ---
    # You can refine this kernel, but this is stable and physical
    U0_hat_x = T_hat_x / (mu * K)
    U0_hat_y = T_hat_y / (mu * K)

    # Depth grid
    z_vals = np.linspace(0, depth, m)

    U = np.zeros((N, N, m, 2), dtype=np.float64)

    for i, z in enumerate(z_vals):
        decay = np.exp(-K * z)
        
        U_hat_x = U0_hat_x * decay
        U_hat_y = U0_hat_y * decay
        
        U[:, :, i, 0] = np.real(np.fft.ifft2(U_hat_x))
        U[:, :, i, 1] = np.real(np.fft.ifft2(U_hat_y))

    return U

def calculate_traction_rk4(depth, surface_traction, E, nu, m):
    N = surface_traction.shape[0]
    dz = depth / (m - 1)
    mu = E / (2 * (1 + nu))
    
    # State contains [Displacement, Traction] -> Shape (N, N, 2, 2) per layer
    # We store the result in the requested (N, N, m, 2) traction array
    T_field = np.zeros((N, N, m, 2))
    U_field = np.zeros((N, N, m, 2))
    
    T_field[:, :, 0, :] = surface_traction

    def get_system_derivatives(U_slice, T_slice):
        # du/dz = T / mu
        du_dz = T_slice / mu
        # dT/dz = -mu * laplacian(U)
        dt_dz = np.zeros_like(T_slice)
        for c in range(2):
            dt_dz[:, :, c] = -mu * (np.gradient(np.gradient(U_slice[:, :, c], axis=0), axis=0) + 
                                    np.gradient(np.gradient(U_slice[:, :, c], axis=1), axis=1))
        return du_dz, dt_dz

    for i in range(m - 1):
        U_n = U_field[:, :, i, :]
        T_n = T_field[:, :, i, :]
        
        # RK4 Steps for coupled system
        du1, dt1 = get_system_derivatives(U_n, T_n)
        du2, dt2 = get_system_derivatives(U_n + 0.5*dz*du1, T_n + 0.5*dz*dt1)
        du3, dt3 = get_system_derivatives(U_n + 0.5*dz*du2, T_n + 0.5*dz*dt2)
        du4, dt4 = get_system_derivatives(U_n + dz*du3, T_n + dz*dt3)
        
        U_field[:, :, i+1, :] = U_n + (dz/6) * (du1 + 2*du2 + 2*du3 + du4)
        T_field[:, :, i+1, :] = T_n + (dz/6) * (dt1 + 2*dt2 + 2*dt3 + dt4)

    # Shift to satisfy U(depth) = 0
    U_bottom = U_field[:, :, -1, :][:, :, np.newaxis, :]
    U_field -= U_bottom
    
    return T_field