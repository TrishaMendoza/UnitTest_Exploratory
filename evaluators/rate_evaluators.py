import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

def is_1d(data):

    if isinstance(data, np.ndarray):
        return data.ndim == 1
    elif isinstance(data, list):
        # If the list's first element is NOT a list or ndarray, treat as 1D
        return len(data) > 0 and not isinstance(data[0], (list, np.ndarray))
    else:
        return False

def compute_comparisons (data_true, data_predicted):

    summary_statistics =[]

    # ------------------ Single Channel ------------------------------
    if is_1d(data_true) and is_1d(data_predicted):
        
        # Compute just for a singular channel
        stat_mu, p_mu = mannwhitneyu(data_true, data_predicted, alternative='two-sided')
        stat_ks, p_ks = ks_2samp(data_true, data_predicted)

        # Determine Pass or Fail
        if p_mu and p_ks < 0.05:
            status = 'Fail'
        elif p_mu and p_ks > 0.05:
            status = 'Pass'
        else:
            status = 'Check'
        
        # Develop Dictionary to append
        summary_statistics.append({'Neuron': 'Neuron 1',
        'Mann-WhitneyU Stat': stat_mu,
        'Mann-WhitneyU P -value': p_mu,
        'KS Stat': stat_ks,
        'KS P -value': p_ks,
        'Pass / Fail': status})

    # ------------------ Single Channel ------------------------------
    else:

        # Detect if input is a list or numpy array
        if isinstance(data_true, list):
            n_neurons = len(data_true)
        elif isinstance(data_true, np.ndarray):
            n_neurons = data_true.shape[0]
        else:
            raise ValueError("Input must be list or 3D numpy array")

        # Now Loop and Compare each of the Neurons
        for neuron_id in range(n_neurons):

            # Extract the different neurons
            if isinstance(data_true, np.ndarray):
                vec_true = data_true[neuron_id, :]
                vec_predicted = data_predicted[neuron_id, :]
            else:
                vec_true = data_true[neuron_id]
                vec_predicted = data_predicted[neuron_id]

            # Compute just for a singular channel
            stat_mu, p_mu = mannwhitneyu(vec_true, vec_predicted)
            stat_ks, p_ks = ks_2samp(vec_true, vec_predicted)

            # Determine Pass or Fail
            if p_mu and p_ks < 0.05:
                status = 'Fail'
            elif p_mu and p_ks > 0.05:
                status = 'Pass'
            else:
                status = 'Check'
        
            # Develop Dictionary to append
            summary_statistics.append({'Neuron': f"Neuron {neuron_id+1}",
            'Mann-WhitneyU Stat': stat_mu,
            'Mann-WhitneyU P -value': p_mu,
            'KS Stat': stat_ks,
            'KS P -value': p_ks,
            'Pass / Fail': status})


    # Develop final dataframe
    summary_statistics_df = pd.DataFrame(summary_statistics)
    return summary_statistics_df

