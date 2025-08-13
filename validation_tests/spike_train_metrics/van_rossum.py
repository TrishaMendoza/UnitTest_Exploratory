import numpy as np
from scipy.signal import convolve


def van_rossum_distance_analysis(spike_train_a, spike_train_b, sample_rate_hz, tau_ms = 10):

    """
    Van Rossum Distance:
    Compute the Van Rossum distance between two spike trains.
    The Van Rossum distance is a measure of the similarity between two spike trains.
   
    Input:
        1. spike_train_a: matrix / vector of spike train(s) [neuron, samples]
        2. spike_train_b: matrix / vector of spike train(s) [neuron, samples]
        3. sample_rate_hz: sampling frequency for data [Hz]
        4. tau_ms: time constant for the exponential kernel [ms]

    Output:
        1. van_rossum_dist: final distance value for each neuron

    """


    # Check if the spike trains are the same length
    if spike_train_a.shape != spike_train_b.shape:
        raise ValueError("Spike matrices trains must be the same length")

    # Ensure it is a 2D input for uniform calculations
    if spike_train_a.ndim == 1:
        spike_train_a = np.atleast_2d(spike_train_a)
    if spike_train_b.ndim == 1:
        spike_train_b = np.atleast_2d(spike_train_b)

    # Kernel Parameter
    tau = tau_ms / 1000
    dt = 1 / sample_rate_hz

    # Set up the kernel for convolution
    kernel_length = int(5 * tau / dt) # how much of the exponential we are extracting so it is not infinite
    kernel_time = np.arange(0, kernel_length) * dt
    kernel = np.exp(-kernel_time / tau)

    # Begin Convolving the different spike trains 
    n_neurons = spike_train_a.shape[0]
    van_rossum_dist = np.zeros(n_neurons)

    for indx in range(n_neurons):

        # Extract the spike trains 
        train_a = spike_train_a[indx, :]
        train_b = spike_train_b[indx, :]

        # Convolve the two spike trains
        convolve_train_a = convolve(train_a, kernel, mode = 'full')
        convolve_train_b = convolve(train_b, kernel, mode = 'full')

        # Calculate the Van Rossum Distance 
        van_rossum_squared = (dt/tau) * np.sum((convolve_train_a - convolve_train_b)**2)

        # Store the final value 
        van_rossum_dist[indx] = np.sqrt(van_rossum_squared)

    return van_rossum_dist