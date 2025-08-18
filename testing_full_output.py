import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from spike_train_metrics.spike_rate import spike_rate_analysis
from spike_train_metrics.interspike_interval import interspike_interval_analysis
from spike_train_metrics.schreiber_similarity import schreiber_similarity_analysis
from spike_train_metrics.van_rossum import van_rossum_distance_analysis
from spike_train_metrics.multi_scale_correlation import run_multiscale_correlation_analysis
from spike_train_metrics.granger_causality import granger_causality_analysis
from behavior_metrics.truth_table_validator import identify_behaviour_patterns
from evaluators.rate_evaluators import compute_comparisons

from load_data import spike_train_true, spike_train_predicted

sns.set_theme()

# ----------------------------------------------------------------------------------------------
# Import the first trial for 
sample_rate = 1000
n_samples = spike_train_true.shape[1]
time_vector = np.arange(0, spike_train_true.shape[1]) #/sampling_rate

# Plot the different neurons
n_neurons = spike_train_true.shape[0]
neuron_labels = ['Neuron A', 'Neuron B', 'Neuron C', 'Neuron D', 'Neuron E']
#for neuron_id in range(n_neurons):
#    plt.subplot(n_neurons, 1, neuron_id+1 )
#    sns.lineplot(x = time_vector, y = spike_train_true[neuron_id,:], label = 'True')
#    sns.lineplot(x = time_vector, y = spike_train_predicted[neuron_id,:], label = 'Predicted')
#    plt.xlabel('Time [ms]')
#    plt.title("Neuron " + neuron_labels[neuron_id])
#plt.show()


# ----------------------------------------------------------------------------------------------
# 1. Spike Rate Testing
spike_rate_true, fano_true = spike_rate_analysis(spike_train_true, sample_rate_hz = 1000, bin_size_ms = 1500, step_size_ms = 50)
spike_rate_predicted, fano_pred = spike_rate_analysis(spike_train_predicted, sample_rate_hz = 1000, bin_size_ms = 1500, step_size_ms = 50)


# Develop Fano Factor Bar Plots
ff_df = pd.DataFrame({
    'Neuron' : neuron_labels + neuron_labels,
    'Fano Factor' : np.append(fano_true, fano_pred),
    'Type' : ['True'] * n_neurons + ['Predicted'] * n_neurons
})

rate_statistics_df = compute_comparisons (spike_rate_true, spike_rate_predicted)
print(rate_statistics_df)


# Plotting spike train results
#for neuron_id in range(n_neurons+1):
#    plt.subplot(2, 3, neuron_id + 1)
#    if neuron_id <= 4:
#        sns.histplot( data = spike_rate_true[neuron_id,:], bins = 5 )
#        sns.histplot( data = spike_rate_predicted[neuron_id,:], bins = 5 )
#        plt.title(neuron_labels[neuron_id] + " Spike Rate")
#    else:
#        sns.barplot(data=ff_df, x='Neuron', y='Fano Factor', hue='Type', palette='pastel')
#        plt.title("Fano Factor")

#plt.show()

# ----------------------------------------------------------------------------------------------
# 2. Interspike Interval 
interspike_true, cv_true = interspike_interval_analysis(spike_train_true, sample_rate_hz = 1000)
interspike_pred, cv_pred = interspike_interval_analysis(spike_train_predicted, sample_rate_hz = 1000)

# Develop CV Bar Plots
cv_df = pd.DataFrame({
    'Neuron' : neuron_labels + neuron_labels,
    'Coefficient of Variation' : np.append(cv_true, cv_pred),
    'Type' : ['True'] * n_neurons + ['Predicted'] * n_neurons
})

isi_statistics_df = compute_comparisons (interspike_true, interspike_pred)
print(isi_statistics_df)

# Plotting spike train results
#for neuron_id in range(n_neurons+1):
    #plt.subplot(2, 3, neuron_id + 1)
    #if neuron_id <= 4:
    #    sns.histplot( data = interspike_true[neuron_id], bins = 20 )
    #    sns.histplot( data = interspike_pred[neuron_id], bins = 20 )
    #    plt.title(neuron_labels[neuron_id] + " ISI")
    #else:
    #    sns.barplot(data=cv_df, x='Neuron', y='Coefficient of Variation', hue='Type', palette='pastel')
    #    plt.title("Coefficient of Variation")
#plt.show()

# ----------------------------------------------------------------------------------------------
# 3. Schreiber Similarity
schreiber_df = schreiber_similarity_analysis(spike_train_true, spike_train_predicted, sample_rate, sigma_ms = 5)
print(schreiber_df)

# ----------------------------------------------------------------------------------------------
# 4. Van Rossum
van_rossum_df = van_rossum_distance_analysis(spike_train_true, spike_train_predicted, sample_rate, tau_ms = 10)
print(van_rossum_df)

# Plot Similarity Metrics
#plt.subplot(1,2,1)
#sns.barplot(data = schreiber_df, x='Neuron', y='SchreiberSimilarity', color='#B5EAD7')
#plt.title('Schreiber Similarity')

#plt.subplot(1,2,2)
#sns.barplot(data = van_rossum_df, x='Neuron', y='VRDistance', color='#A2D2FF')
#plt.title('Van Rossum Distance')
#plt.show()

# ----------------------------------------------------------------------------------------------
# 5. Multi Scale Correlation
correlation_results_df, correlation_results, sigma_vals = run_multiscale_correlation_analysis(spike_train_true, spike_train_predicted, sample_rate)
print(correlation_results_df)


#for neuron_id in range(n_neurons):
#    plt.subplot(2, 3, neuron_id + 1)
#    sns.lineplot( x = sigma_vals, y = correlation_results[neuron_id] )
#    plt.title(neuron_labels[neuron_id] + " Correlation Plots")
#    plt.xlabel('Sigma Values')
#    plt.ylabel('Correlation')
#plt.show()

# ------------------------- Section just for behaviour testing the function -----------------
# Truth Table with Expected Input and Expected Output
test_truth_table_df = pd.DataFrame({
    "Input A": [1,1,0],
    "Input B": [0,1,1],
    "Ouput C": [1,0,1],
})
performance_df = identify_behaviour_patterns(test_truth_table_df, spike_train_predicted[:1,:], spike_train_predicted[4,:])
print(performance_df)

# ----------------------------------------------------------------------------------------------
# 6. Granger Causality
#granger_lag_true, granger_ftest_true, granger_pval_true = granger_causality_analysis(spike_train_true, sample_rate, max_lag = 10, bin_size_ms = 5)
#granger_lag_pred, granger_ftest_pred, granger_pval_pred = granger_causality_analysis(spike_train_predicted, sample_rate, max_lag = 10, bin_size_ms = 5)

# Create DataFrame with labels
#granger_lag_true_df = pd.DataFrame(granger_lag_true, index = neuron_labels, columns = neuron_labels)
#granger_lag_pred_df = pd.DataFrame(granger_lag_pred, index = neuron_labels, columns = neuron_labels)

#ftest_true_df = pd.DataFrame(granger_ftest_true, index = neuron_labels, columns = neuron_labels)
#ftest_pred_df = pd.DataFrame(granger_ftest_pred, index = neuron_labels, columns = neuron_labels)

#pval_true_df = pd.DataFrame(granger_pval_true, index = neuron_labels, columns = neuron_labels)
#pval_pred_df = pd.DataFrame(granger_pval_pred, index = neuron_labels, columns = neuron_labels)

  



