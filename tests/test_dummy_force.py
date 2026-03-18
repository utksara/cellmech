import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from cellmech import *


def test_dummy():

    image_name = "handdrawn_twocells"
    # open microfscopic image of cell and convert it into matrix
    image_matrix = np.array(Image.open(f'images/real_cells/{image_name}.png').convert('L'))
    image_dim = np.shape(image_matrix)

    # define all the parameters that are required to calculata traction forces
    cellmechparams = CellMechParameters({
        "p" : 0.5,
        "E" : 10e3,
        "pi" : np.pi,
        "N" : 50,
        "width" :2,
        "pixel_size" : 1e-7 
    })

    '''
    first we will a dummy force field using known function (symmetric gausian) to calcualte displacement field, then we will 
    use generated dispclement field to retrieve original force field so see if calculations match 
    '''

    force_points_list, updated_image = detect_shapes(image_matrix, detection_threshold = 0.7)

    x_axis = np.linspace(-1, 1, updated_image.shape[0])
    y_axis = np.linspace(-1, 1, updated_image.shape[1])

    print("len ", len(force_points_list))
    # calculation of force field using gaussian
    image_dims = image_matrix.shape
    force_field, force_location  = calculate_dummy_force(force_points_list, cellmechparams)
    sns.heatmap(force_location)
    fig = plot_vector_field(force_field, "vector map of force field")
    
    sns.heatmap(force_location)
    plot_vector_field(force_field, "vector map of force field")
    plt.show()