import numpy as np
# traction force kernel steps

input_scalar = ["young", "pois", "pixelsize"]
input_vector = ["X", "Y" ,"ux", "uy"]

measureables = ["Pre stress", "displacement field", ]

# steps

pois = 1
pi = 1
young = 1
X = np.zeroes()
Y = np.zeroes()

# a1   = (1.0 + pois) * (1.0 - pois) / (pi * young)
# b1   = (1.0 + pois) * pois / (pi * young)
# c1   = (1.0 + pois) * pois / (pi * young)

# xv  = (X(:,1) - mean(Y(:,1))) * pixelsize  
# yv  = (X(:,2) - mean(Y(:,2))) * pixelsize
# uxv = (ux(:,3)) * pixelsize
# uyv = (uy(:,4)) * pixelsize
