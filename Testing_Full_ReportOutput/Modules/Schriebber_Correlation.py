import numpy as np 
import scipy.io as sio
from scipy.signal import convolve

def calculate_correlation(spike_train_a, spike_train_b, sigma, fs):

    # Develop the Gaussian Kernel
    dt = 1 / fs
    kernel_size = (6 * sigma / dt).astype(int) # 6 ensures we are covering +/- 3 standard deviations

    # Check if kernel is odd so either side of gaussian is event
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Kernel time vector
    kernel_t = np.linspace(-3 * sigma, 3 * sigma, kernel_size) 
    gaussian_kernel = (1 / (np.sqrt(2 * np.pi) * sigma) ) * np.exp(-kernel_t**2 / (2 * sigma**2))

    # Convolve with your spike trains
    convolve_train_a = convolve(spike_train_a, gaussian_kernel, mode = 'same')
    convolve_train_b = convolve(spike_train_b, gaussian_kernel, mode = 'same')

    # Calculate the Correlation between the signals
    numerator_val = np.mean(convolve_train_a*convolve_train_b)
    denominator_val = np.sqrt(np.mean(convolve_train_a*convolve_train_a)*np.mean(convolve_train_b*convolve_train_b))

    return numerator_val / denominator_val