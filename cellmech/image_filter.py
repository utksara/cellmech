import numpy as np
from PIL import Image
import seaborn as sns

image_matrix = np.array(Image.open('images/img1.png').convert('L'))

sns.heatmap(image_matrix)
