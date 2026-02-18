from cellmech import plot_vector_field, calcualte_displacement, CellMechParameters, calculate_analytical_displacement
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import numpy as np

def test_point_force():
    
    N = 50
    v = 0.5
    E = 1e3
    width = 0.1
    dx = width/N
    cellmechparams = CellMechParameters({
        "p" : v,
        "E" : E,
        "pi" : np.pi,
        "N" : N,
        "width" : 0.1,
        "pixel_size" : 1e-7
    })


    force_locations = [np.array([0.025, 0.025]), np.array([0.025, -0.01]), np.array([0.01, -0.025]), np.array([-0.02, -0.02])]
    force_values = [np.array([1, 0]), np.array([1/np.sqrt(2), -1/np.sqrt(2)]), np.array([0, 1]), np.array([np.sqrt(3)/2, -1/2])]
    
    for force_location, force in zip(force_locations, force_values):
        
        fmag = np.sqrt(force[0] + force[1])
        print("dimensionless displacement : ",  fmag / (np.pi * E * width))
        print(f'\n{"="*3} Force location from center of substrate in cm : {100*force_location} {"="*3}')
        force_position = np.array([int((N - 1) * (force_location[0]  + width/2)/(width)), int((N - 1) * (force_location[1]  + width/2)/(width))])
        print(f'{"="*3} Force location inedex on array : {force_position} {"="*3}')
        force_field = np.zeros((N, N, 2))
        force_field[force_position[0], force_position[1], :] = force
        
        theoretic_disp = calculate_analytical_displacement(force_field, force_position, cellmechparams)
        fttc_calc_disp = calcualte_displacement(force_field/dx**2, cellmechparams)
        # tn_method_disp = calcualte_displacement(force_field, cellmechparams, "tn")
        # plot_vector_field(theoretic_disp, "Theoretical displacement field")
        # plot_vector_field(fttc_calc_disp, "FTTC calcul displacement field")
        # plot_vector_field (tn_method_disp, "tensor netw displacement field")
        print(" max abs theoretic_disp : ", np.max(abs(theoretic_disp)), " , max abs fttc_calc_disp : ", np.max(abs(fttc_calc_disp)))
        print("error  = ", 100 * np.max(abs(theoretic_disp - fttc_calc_disp))/np.max(abs(theoretic_disp)))
