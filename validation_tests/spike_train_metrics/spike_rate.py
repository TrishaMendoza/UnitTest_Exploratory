import numpy as np

def spike_rate_analysis(spike_train, 
    sample_rate_hz , 
    bin_size_ms = 50,  # [ms]
    step_size_ms = 25, # [ms]
    ):

    """
    Spike Rate Function:
    This function will take in either a single or multiple spike trains and 
    compute the number of spikes per a window of time [ms]

    Input: 
        1. spike_tain: Spike train data [channels, samples]
        2. bin_size_ms: window size for calculating the rate
        3. step_size_ms: how much to step over in time 
        4. sample_rate_hz: sampling frequency for the data provided [Hz]

    Output:
        1. Spike Rate Vector or Matrix [channels, rate]
        2. Fano Factor: calculating the fano factor from the spike rate

    """
    # Ensure it is a 2D input for uniform calculations
    if spike_train.ndim == 1:
        spike_train = np.atleast_2d(spike_train)

    # Convert time to samples 
    bin_size_samples = int( (bin_size_ms / 1000) * sample_rate_hz)
    step_size_samples = int( (step_size_ms / 1000) * sample_rate_hz)

    # Determine # of windows to slide over
    spike_train_duration = spike_train.shape[1]
    if step_size_ms > 0:
        n_steps = int( (spike_train_duration - bin_size_samples) / step_size_samples ) + 1
    else:
        step_size_samples = bin_size_samples
        n_steps = int(spike_train_duration / bin_size_samples)

    # Initialize Storage for spike rate
    spike_rate_matrix = np.zeros([ spike_train.shape[0] , n_steps]) 
    fano_factor = []

    # Begin looping through the spike train
    for indx in range(n_steps):

        # Initialize start and stop indices
        if indx == 0:
            start_indx = indx
            stop_indx = bin_size_samples
        else:
            start_indx = start_indx + step_size_samples
            stop_indx = start_indx + bin_size_samples

        # Extract the Data and Compute the Rate 
        temp_rate = np.sum(spike_train[:, start_indx : stop_indx], axis = 1) / bin_size_ms
        spike_rate_matrix[ :, indx ] = temp_rate

    # Calculate the Fano Factor
    fano_factor = np.var(spike_rate_matrix, axis = 1)/np.mean(spike_rate_matrix, axis = 1)

    return spike_rate_matrix, fano_factor
