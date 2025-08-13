import numpy as np
from scipy.signal import convolve

# ----------------------------- prepare the data for analysis -------------------------------------
def run_multiscale_correlation_analysis(spike_train_matrix_a, spike_train_matrix_b, sample_rate_hz):

    """
    Run multiscale correlation analysis - simply prepares the analysis for the correlation across multiple
    neurons if provided - or can run a single vector as well

    Inputs:
        1. spike_train_matrix_a: spike train matrix or vector [neurons, samples]
        2. spike_train_matrix_b: spike train matrix or vector [neurons, samples]
        3. sample_rate_hz: sampling rate for the data [hz]

    Output:
        1. correlation_results: correlation curves for different neurons [neurons, correlatio values]
    """
    
    # Check if the spike trains are the same length
    if spike_train_matrix_a.shape != spike_train_matrix_b.shape:
        raise ValueError("Spike matrices trains must be the same length")

    # Check if we recieved a vector or a matrix
    if spike_train_matrix_a.ndim > 1:

        # Prepare the correlation storage 
        correlation_results = []
        n_neurons = spike_train_matrix_a.shape[0]

        # Loop through the different neuron pairs 
        for neuron_id in range(n_neurons):
            train_a = spike_train_matrix_a[neuron_id, :]
            train_b = spike_train_matrix_b[neuron_id, :]

            # Run Correlation Analysis
            temp_corr, sigma_vals = multi_scale_correlation(train_a, train_b, sample_rate_hz)
            correlation_results.append(temp_corr)
    else:

        # If there is only a vector provided
        correlation_results, sigma_vals = multi_scale_correlation(spike_train_matrix_a, spike_train_matrix_b, sample_rate_hz)
    
    return correlation_results, sigma_vals
        

# ----------------------------- run multi scale correlation -------------------------------------
def multi_scale_correlation(spike_train_a, spike_train_b, sample_rate_hz):

    """
    Multi Scale Correlation:
    Calculating the correlation across multiple sigmas
    """

    # Range of Sigma Values
    sigma_vals = np.arange(1, 100, 1) / 1000
    correlation_vals = np.zeros(len(sigma_vals))

    # Develop the Gaussian Kernel
    dt = 1 / sample_rate_hz

    for indx, sigma in enumerate(sigma_vals):

        # Develop Gaussian Kernel
        kernel_size = int(6 * sigma / dt) # 6 ensures we are covering +/- 3 standard deviations
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

        # Store the Correlation Values
        correlation_vals[indx] = numerator_val / denominator_val
    

    return correlation_vals, sigma_vals
