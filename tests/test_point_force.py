from cellmech import plot_vector_field, calcualte_displacement, CellMechParameters
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def calculate_analytical_displacment(force, force_position, point_position, v, E):
    v1 = force_position - point_position
    v1 = v1/np.sqrt(v1[0]**2 + v1[1]**2 + 1e-9)
    v2 = force/np.sqrt(force[0]**2 + force[1]**2)    
    r = force_position - point_position
    rmag = np.sqrt(r[0]**2 + r[1]**2) + 1e-3
    fmag = np.sqrt(force[0]**2 + force[1]**2)
    costheta = np.inner(v1, v2)
    sintheta = np.cross(np.array([v1[0], v1[1], 0]), np.array([v2[0], v2[1], 0]))[2]
    
    print(costheta, sintheta, costheta**2 + sintheta**2) 
    force_coeff = (1 - v**2)*fmag/(rmag * np.pi * E )
    ux =  force_coeff * costheta
    uy = -force_coeff * sintheta
    return np.array([ux, uy])


force = np.array([1, 0])
force_position = np.array([0, 0])

N = 20
traction_force = np.zeros((N, N, 2))
v = 0.5
E = 1e3

dx = 1/N
for i in range(0, N):
    for j in range(0, N):
        point_position = dx*np.array([(i - int(N/2)), (j - int(N/2))])
        traction_force[i,j, :] = calculate_analytical_displacment(force, force_position, point_position, v, E)


cellmechparams = CellMechParameters({
    "p" : v,
    "E" : E,
    "pi" : np.pi,
    "N" : N,
    "width" :1,
    "pixel_size" : 1e-7
})

force_field = np.zeros((N, N, 2))

force_field[force_position[0], force_position[1], 0] = 1
tfm_force = calcualte_displacement(force_field, cellmechparams)

print("error ", tfm_force - traction_force)
# plot_vector_field(traction_force)
