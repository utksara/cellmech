from cellmech import calculate_displacement, CellMechParameters, plot_vector_field, tensor_contraction, getG
import numpy as np
import os

N = 20
v = 0.5
E = 1e3
dx = 1/N

cellmechparams = CellMechParameters({
    "p": v,
    "E": E,
    "pi": np.pi,
    "N": N,
    "width": 1,
    "pixel_size": 1e-7
})


def plot_vector_field_test(vecfield, title):
    if int(os.environ.get("ENABLE_VISUAL_TESTING", False)):
        plot_vector_field(vecfield, title)


n = 21
disp = np.zeros((n, n, 2))
force_field = np.zeros((n, n, 2))
force_field[int(1*n/4), int(1*n/4), 0] = -1
displacement = calculate_displacement(force_field, cellmechparams, "tn")


def getG1(v: np.ndarray[tuple[int]]):
    x = v[0]
    y = v[1]
    r = np.sqrt(x**2 + y**2)
    if r <= 1e-3:
        return np.zeros((2, 2))
    return np.array([[1/r, 0], [0, 1/r]])


def getG2(v: np.ndarray[tuple[int]]):
    return getG(v, cellmechparams.params)


def _matrix_2x2_inverse(A: np.ndarray[tuple[tuple[int]]]):
    factor = 1/(A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0])
    M = np.array([[A[1, 1], -A[0, 1]],
                  [-A[1, 0], A[0, 0]]])
    return factor*M


def test_matrix_inverse():

    tolerence = 1e-20
    A = np.array([[1, 2], [-2, 1]])
    assert abs(np.sum(np.linalg.inv(A) - _matrix_2x2_inverse(A))) / \
        4 <= tolerence

    A = np.array([[1, 2], [-1.5, 1]])
    assert abs(np.sum(np.linalg.inv(A) - _matrix_2x2_inverse(A))) / \
        4 <= tolerence

    A = np.array([[1, 2], [1, 1.5]])
    assert abs(np.sum(np.linalg.inv(A) - _matrix_2x2_inverse(A))) / \
        4 <= tolerence


def hard_coded_contraction(gten, uten):
    nx = gten.shape[0]
    ny = gten.shape[1]

    assert gten.shape[3] == uten.shape[2], "unequal matrices"

    vten = np.zeros(uten.shape[2])
    for l in range(0, nx):
        for m in range(0, ny):
            vten = vten + gten[l, m, :, :] @ uten[l, m, :]
    return vten


def normalized_grid_distance(r, c, N):
    p1 = (min(r[0] + 1, N - 1), r[1])
    p2 = (r[0], min(r[1] + 1, N - 1))
    p3 = (max(r[0] - 1, 0), r[1])
    p4 = (r[0], max(r[1] - 1, 0))

    average_dist = ((p1[0] - c[0])**2 + (p1[1] - c[1])**2 +
                    (p2[0] - c[0])**2 + (p2[1] - c[1])**2 +
                    (p3[0] - c[0])**2 + (p3[1] - c[1])**2 +
                    (p4[0] - c[0])**2 + (p4[1] - c[1])**2 +
                    (r[0] - c[0])**2 + (r[1] - c[1])**2) / 5
    return np.sqrt(average_dist)


def test_tensor_contract():
    N = 10
    tolerence = 1/N
    Gten = np.zeros((2*N, 2*N, 2, 2))
    uten = np.zeros((N, N, 2))
    uten[int((N-1)/2), int((N-1)/2), 0] = 0
    uten[int((N-1)/2), int((N-1)/2), 1] = 0

    for i in range(0, 2*N):
        for j in range(0, 2*N):
            u = np.array([i - int((2*N - 1)/2), j - int((2*N - 1)/2)])
            Gten[i, j, :, :] = getG2(u)

    for i in range(0, N):
        for j in range(0, N):
            if (normalized_grid_distance((i, j), ((N-1)/2, (N-1)/2), N) > N/4):
                u = np.array([i - (N-1)/2, j - (N-1)/2])
                if np.linalg.norm(u) <= tolerence:
                    uten[i, j, :] = np.zeros((2, 2))
                    break
                u = u/(np.linalg.norm(u))
                cx = (N-1)/2 + N/4*u[0]
                cy = (N-1)/2 + N/4*u[1]
                v = np.array([cx - i, cy - j])
                d = normalized_grid_distance((i, j), (cx, cy), N)
                uten[i, j, :] = np.exp(-d**2) * v/(np.linalg.norm(v))

    plot_vector_field_test(uten, "uten")
    gten = Gten[int((2*N-1)/2) - int((N-1)/2): int((2*N-1)/2) - int((N-1)/2) + N,
                int((2*N-1)/2) - int((N-1)/2): int((2*N-1)/2) - int((N-1)/2) + N, :, :]
    print(
        f'\n ==== tensor contraction ==== \n {tensor_contraction(gten, uten)}')
    print(
        f'\n ==== hardco contraction ==== \n {hard_coded_contraction(gten, uten)}')
    assert abs(np.sum(hard_coded_contraction(gten, uten) - tensor_contraction(gten, uten))
               )/(N**2) <= tolerence, "contraction frmo tn and harc code do not match"

    vten = np.zeros((N, N, 2))
    for i in range(0, N):
        for j in range(0, N):
            gten = Gten[int((2*N-1)/2) - i: int((2*N-1)/2) - i + N,
                        int((2*N-1)/2) - j: int((2*N-1)/2) - j + N, :, :]
            vten[i, j, :] = hard_coded_contraction(gten, uten)
            assert abs(np.sum(hard_coded_contraction(gten, uten) - tensor_contraction(gten, uten))
                       )/(N**2) <= tolerence, "contraction frmo tn and harc code do not match"

    plot_vector_field_test(vten, "vten")


def test_simple_forces():
    N = 10
    tolerence = 1/N
    Gten = np.zeros((2*N, 2*N, 2, 2))
    uten = np.zeros((N, N, 2))
    uten[int((N-1)/2), int((N-1)/2), 0] = 0
    uten[int((N-1)/2), int((N-1)/2), 1] = 0

    for i in range(0, 2*N):
        for j in range(0, 2*N):
            u = np.array([i - int((2*N - 1)/2), j - int((2*N - 1)/2)])
            Gten[i, j, :, :] = getG2(u)

    uten[int(N/2), int(N/2), :] = np.array([1, 0])
    plot_vector_field_test(uten, "uten")
    vten = np.zeros((N, N, 2))
    for i in range(0, N):
        for j in range(0, N):
            gten = Gten[int((2*N-1)/2) - i: int((2*N-1)/2) - i + N,
                        int((2*N-1)/2) - j: int((2*N-1)/2) - j + N, :, :]
            vten[i, j, :] = tensor_contraction(gten, uten)

    uten[int(N/2), int(N/2), :] = np.array([-1, 0])
    plot_vector_field_test(uten, "uten")
    vten = np.zeros((N, N, 2))
    for i in range(0, N):
        for j in range(0, N):
            gten = Gten[int((2*N-1)/2) - i: int((2*N-1)/2) - i + N,
                        int((2*N-1)/2) - j: int((2*N-1)/2) - j + N, :, :]
            vten[i, j, :] = hard_coded_contraction(gten, uten)
            assert abs(np.sum(hard_coded_contraction(gten, uten) - tensor_contraction(gten, uten))
                       )/(N**2) <= tolerence, "contraction frmo tn and harc code do not match"

    plot_vector_field_test(vten, "vten")

    uten[int(N/2), int(N/2), :] = np.array([1, -1])
    plot_vector_field_test(uten, "uten")
    vten = np.zeros((N, N, 2))
    for i in range(0, N):
        for j in range(0, N):
            gten = Gten[int((2*N-1)/2) - i: int((2*N-1)/2) - i + N,
                        int((2*N-1)/2) - j: int((2*N-1)/2) - j + N, :, :]
            vten[i, j, :] = hard_coded_contraction(gten, uten)
            assert abs(np.sum(hard_coded_contraction(gten, uten) - tensor_contraction(gten, uten))
                       )/(N**2) <= tolerence, "contraction frmo tn and harc code do not match"

    plot_vector_field_test(vten, "vten")
