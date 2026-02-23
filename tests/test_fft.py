import numpy as np
from scipy.signal import convolve
import seaborn as sns
import matplotlib.pyplot as plt
import os

# definition of rectangular pulse function
def rect_pulse(x, a):
    return np.where(np.abs(x) <= a/2, 1, 0)

# definition of gaussian functoin
def gaussian(x, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma)**2)

# function 1 : return convolution function between two function based on their discrete values calculated over a unform grid
def convolution_discrete(f, g, dx):
    return convolve(f, g, mode='same') * dx

# function 2 : return convolution function based on analytical formula between gaussian and rectangular pulse function
def convolution_analytical(x, a, sigma):
    from scipy.special import erf
    return (erf((x + a/2) / (np.sqrt(2) * sigma)) - erf((x - a/2) / (np.sqrt(2) * sigma))) / 2

# function 3 : calculate fft of a given function based basen on their discrete values calculated over a unform grid
def fft_discrete(f, dx):
    return np.fft.fft(f) * dx

# function 4 : calculate convolution value using convolution theorem (taking fft of indivudual functions, multiplying them and taking ifft) from discrete values calculated over a unform grid
def convolution_fft(f, g, dx):
    F = fft_discrete(f, dx)
    G = fft_discrete(g, dx)
    return np.fft.ifft(F * G) / dx

# definition of 2d rectangular pulse function
def rect_pulse_2d(x, y, a):
    return np.where((np.abs(x) <= a/2) & (np.abs(y) <= a/2), 1, 0)

# definition of 2d gaussian functoin
def gaussian_2d(x, y, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

def convolution2_analytical(x, y, a, sigma):
    from scipy.special import erf
    term_x = (erf((x + a/2) / (np.sqrt(2) * sigma)) - erf((x - a/2) / (np.sqrt(2) * sigma))) / 2
    term_y = (erf((y + a/2) / (np.sqrt(2) * sigma)) - erf((y - a/2) / (np.sqrt(2) * sigma))) / 2
    return term_x * term_y

# function 5 : calculate fft of a given function based basen on their discrete values calculated over a unform grid
def fft2_discrete(f, dx):
    return np.fft.fft2(f) * dx**2

# function 6 : return convolution function between two function based on their discrete values calculated over a unform grid
def convolution2_discrete(f, g, dx):
    return convolve(f, g, mode='same') * dx**2

# test 1 : take discrete value of gaussian function over a selected unform grid, apply fft and followed by ifft, the difference between the value of original function and fft->ifft should not me more than 1 percent
def test_fft_ifft():
    L = 20
    n = 1024
    x = np.linspace(-L/2, L/2, n)
    dx = L / n
    sigma = 0.5
    g = gaussian(x, sigma)

    # Apply FFT and then IFFT
    g_fft = fft_discrete(g, dx)
    g_ifft = np.fft.ifft(g_fft) / dx

    # Check the difference
    assert np.allclose(g, g_ifft.real, atol=1e-2)

# test 2 : calculate convolution of gaussian and rectangular pulse function using function 1 and functon 2 and error between the two should not be more than 10 percent
def test_convolution():
    L = 20
    n = 1024
    x = np.linspace(-L/2, L/2, n)
    dx = L / n
    a = 2
    sigma = 0.5

    # Discrete functions
    g = gaussian(x, sigma)
    r = rect_pulse(x, a)

    # Discrete convolution
    conv_d = convolution_discrete(g, r, dx)

    # Analytical convolution
    conv_a = convolution_analytical(x, a, sigma)

    # Check the difference
    assert np.allclose(conv_d, conv_a, atol=0.1)

# test 3 : repeat test 1 but with 2-d gaussian function
def test_fft_ifft_2d():
    L = 20
    n = 128 # smaller n for 2d to keep it fast
    x = np.linspace(-L/2, L/2, n)
    y = np.linspace(-L/2, L/2, n)
    dx = L / n
    xx, yy = np.meshgrid(x, y)
    sigma = 0.5
    g = gaussian_2d(xx, yy, sigma)

    # Apply FFT and then IFFT
    g_fft = fft2_discrete(g, dx)
    g_ifft = np.fft.ifft2(g_fft) / dx**2

    # Check the difference
    assert np.allclose(g, g_ifft.real, atol=1e-2)

# test 4 : repeat test 2 but with 2-d gaussian function and 2-d rectangular pulse function
def test_convolution_2d():
    L = 20
    n = 256
    x = np.linspace(-L/2, L/2, n)
    y = np.linspace(-L/2, L/2, n)
    dx = L / n
    xx, yy = np.meshgrid(x, y)
    a = 2
    sigma = 0.5

    # Discrete functions
    g = gaussian_2d(xx, yy, sigma)
    r = rect_pulse_2d(xx, yy, a)

    # Discrete convolution
    conv_d = convolution2_discrete(g, r, dx)
    if int(os.environ.get("ENABLE_VISUAL_TESTING", False)):
        sns.heatmap(conv_d)
        plt.show()
    
    # Analytical convolution
    conv_a = convolution2_analytical(xx, yy, a, sigma)
    print("max conv_a, max conv_d ", np.max(conv_a), np.max(conv_d))
    if int(os.environ.get("ENABLE_VISUAL_TESTING", False)):
        sns.heatmap(conv_a)
        plt.show()
    
    # Check the difference
    assert np.allclose(conv_d, conv_a, atol=0.1)
