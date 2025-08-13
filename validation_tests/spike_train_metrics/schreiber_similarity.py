import numpy as np
import pandas as pd
from scipy.signal import convolve


def schreiber_similarity_analysis(spike_train_a, spike_train_b, sample_rate_hz, sigma_ms = 10):
    """
    Schreiber Similarity:
    convolve the different spike trains with a guassian and compute 
    schrieber similarity between neurons - tells you how similar the two spike trains are 

    Input:
        1. spike_train_a:
        2. spike_train_b:
        3. sample_rate_hz:
        4. sigma_ms: 
    
    Ouput:
        1. schrieber_sim:

    """
    # Check if the spike trains are the same length
    if spike_train_a.shape != spike_train_b.shape:
        raise ValueError("Spike matrices trains must be the same length")

    # Ensure it is a 2D input for uniform calculations
    if spike_train_a.ndim == 1:
        spike_train_a = np.atleast_2d(spike_train_a)
    if spike_train_b.ndim == 1:
        spike_train_b = np.atleast_2d(spike_train_b)


    # Change ms into s
    sigma = sigma_ms / 1000
    dt = 1 / sample_rate_hz 

    # Develop the gaussian kernel for convolution
    kernel_size = int(6 * sigma /dt) # Covering +/-3 standard deviations
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel_t = np.linspace(-3 * sigma, 3 * sigma, kernel_size) 
    gaussian_kernel = (1 / (np.sqrt(2 * np.pi) * sigma) ) * np.exp(-kernel_t**2 / (2 * sigma**2))

    # Begin Convolving the different spike trains 
    n_neurons = spike_train_a.shape[0]
    schreiber_sim = []

    for indx in range(n_neurons):

        # Extract the spike trains 
        train_a = spike_train_a[indx, :]
        train_b = spike_train_b[indx, :]

        # Convolve the two spike trains
        convolve_train_a = convolve(train_a, gaussian_kernel, mode = 'same')
        convolve_train_b = convolve(train_b, gaussian_kernel, mode = 'same')

        # Calculate the Schrieber Similarity 
        numerator_val = np.mean(convolve_train_a * convolve_train_b)
        denominator_val = np.sqrt(np.mean(convolve_train_a * convolve_train_a) * np.mean(convolve_train_b * convolve_train_b))

        if numerator_val / denominator_val > 0.8:
            status = 'pass'
        else:
            status = 'fail'

        # Store the final value 
        schreiber_sim.append({'Neuron': f"Neuron {indx+1}",
        'SchreiberSimilarity': numerator_val / denominator_val,
        'Pass / Fail': status})

    
    schreiber_sim_df = pd.DataFrame(schreiber_sim)
    return schreiber_sim_df

