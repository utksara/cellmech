import numpy as np

N = 1000

U = np.array([
    [[[1, 0], [0, 1]], [[1, 0], [0, 1]] ], 
    [[[1, 0], [0, 1]], [[1, 0], [0, 1]] ]
    ]
    )
# U = np.ones(N)
# A = np.kron(U, I)

v = np.array([
    [[0 , 0], [0 , 1]],
    [[1 , 0], [1 , 1]]
])
# u = np.kron(U, v)

def get_force_field(G_sub_matrix, force_field):
    S = np.einsum("lmij,lmj->i", G_sub_matrix, force_field)
    return S
print(S)