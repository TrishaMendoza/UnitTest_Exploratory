import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# --------------------------------- run granger analysis -----------------------------------
def granger_causality_analysis(spike_train_matrix, sample_rate_hz, max_lag = 10, bin_size_ms = 5):
    """
    Granger Causality:
    Compute the Granger causality between two spike trains.
    The Granger causality is a measure of the causality between two spike trains.
    It is defined as the ratio of the variance of the predicted spike train to the variance of the actual spike train.


    Input:
        1. spike_train: matrix / vector of spike train(s) [neuron, samples]
        3. sample_rate_hz: sampling frequency for data [Hz]
        4. bin_size_ms: size of the bin for the Granger causality [ms]

    Output:
        1. granger_causality: final causality value for each neuron
    """

    # ------------------------------------------------------------

    # Prepare to bin the data to develop continuous time series
    bin_size = int((bin_size_ms / 1000) * sample_rate_hz)
    n_bins = int(spike_train_matrix.shape[1] / bin_size)

    # Bin the data
    spike_train_binned = np.zeros((spike_train_matrix.shape[0], n_bins))

    # ------------------------------------------------------------
    # Begin Binning time series
    for indx in range(n_bins):

        # Get the start and stop indices
        start_indx = indx * bin_size
        stop_indx = start_indx + bin_size

        # Bin the data
        spike_train_binned[:, indx] = np.sum(spike_train_matrix[:, start_indx : stop_indx], axis = 1)
        
    
    # ------------------------------------------------------------
    # Initialize Storage for the Granger Causality
    n_neurons = spike_train_matrix.shape[0]
    granger_lag = np.zeros((n_neurons, n_neurons))
    granger_ftest = np.zeros((n_neurons, n_neurons))
    granger_pVals = np.zeros((n_neurons, n_neurons))

    # Begin Looping through all possible neurons
    for neuron_a in range(n_neurons):

        # Extract First Spike Train
        binned_train_a =  spike_train_binned[neuron_a, :]

        for neuron_b in range(n_neurons):
            
            # Extract Second Spike Train
            binned_train_b = spike_train_binned[neuron_b, :]

            # Return the best lag statistics
            if neuron_a == neuron_b:

                # Store as a nan value
                granger_lag[neuron_a, neuron_b] = np.nan
                granger_ftest[neuron_a, neuron_b] = np.nan
                granger_pVals[neuron_a, neuron_b] = np.nan
            else:
                # Prepare Dataframe + Run Granger Causality
                #print(f"Testing Neuron {neuron_a} -> {neuron_b}")
                granger_ab = pd.DataFrame({'B': binned_train_b, 'A': binned_train_a})
                results_ab = grangercausalitytests(granger_ab[['B','A']], maxlag = max_lag)
                best_lag, f_stat, p_val = extract_best_granger_stats(results_ab)

                # Store the final results
                granger_lag[neuron_a, neuron_b] = best_lag
                granger_ftest[neuron_a, neuron_b] = f_stat
                granger_pVals[neuron_a, neuron_b] = p_val


    return granger_lag, granger_ftest, granger_pVals


# --------------------------------- extract the lag with the lowest p value ----------------------------------
def extract_best_granger_stats(result_dict):
    """
    Get the lag with the lowest p_value - return the correspinding p value, f-test, and lag
    """

    # Extract the Lag with the lowest p value
    best_lag = min(result_dict, key = lambda lag: result_dict[lag][0]['ssr_ftest'][1])
    f_stat, p_val = result_dict[best_lag][0]['ssr_ftest'][:2]

    return best_lag, f_stat, p_val




