import numpy as np

def interspike_interval_analysis(spike_train, sample_rate_hz):
    """
    Interspike Interval Function:
    This function will take the spike train and calculate the interspike 
    interval for each neuron

    Input:
        1. spike_train: binary spike train vector or matrix [neuron, samples]
        2. sample_rate_hz: sampling frequency for data [Hz]
    
    Output
        1. interspike_matrix: interspike intervals for each neuron
        2. coefficient of variation: coefficient of variation (variability of spike timings)

    """
    # Ensure it is a 2D input for uniform calculations
    if spike_train.ndim == 1:
        spike_train = np.atleast_2d(spike_train)

    # Develop a time vector in [ms] 
    n_samples = spike_train.shape[1]
    time_vector = ( np.arange(n_samples) / sample_rate_hz ) * 1000

    # Initialize Storage
    interspike_matrix = []
    coefficient_variation = np.zeros((spike_train.shape[0], 1))

    # Loop through the matrix
    for indx, train in enumerate(spike_train):

        # Calculate interspike interval
        spike_indx = np.where(train == 1)[0]
        spike_time = time_vector[spike_indx]
        isi_temp = np.diff(spike_time)

        # Calculate coefficient of variation + append final results
        interspike_matrix.append(isi_temp)
        if len(isi_temp) > 0:
            coefficient_variation[indx] = np.round(np.std(isi_temp)/np.mean(isi_temp), decimals = 2)
        else:
            coefficient_variation[indx] = np.nan
            


    return interspike_matrix, coefficient_variation







    







