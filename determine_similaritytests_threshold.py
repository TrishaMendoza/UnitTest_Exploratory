import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from spike_train_metrics.schreiber_similarity import schreiber_similarity_analysis
from spike_train_metrics.van_rossum import van_rossum_distance_analysis
from spike_train_metrics.multi_scale_correlation import run_multiscale_correlation_analysis

from load_data import spike_train_true

# --------------------------- Function for Simulated Data ---------------------------------

# Develop Function that will jitter the spikes within spike train
def jitter_spike_train(spike_train, sample_rate, bin_size_ms = 50):

    # Convert into samples
    bin_size = int((bin_size_ms / 1000) * sample_rate)

    # Loop through spike train windows and re-assign times
    n_bins= int( len(spike_train) / bin_size )
    simulated_spike_train = np.zeros(spike_train.shape)

    for bin_id in range(n_bins):

        # Indicate Start and Stop Indices
        start_indx = int(bin_id * bin_size)
        stop_indx = int(start_indx + bin_size)

        # Get random spike counts
        spike_count = int(np.sum(spike_train[start_indx:stop_indx]))

        if spike_count > 0:
            # Now insert the spikes into the new spike train
            insert_spikes = np.random.randint(low = start_indx, high = stop_indx, size = spike_count)
            simulated_spike_train[insert_spikes] = 1

    return simulated_spike_train



# ------------------------------- Run the Analysis --------------------------------------
# Send Spike Train into 
sample_rate = 1000
time_vector = np.arange(0, spike_train_true.shape[1]) 

# Similarity Data Frame + Simulation Testing
n_simulations = 3000
similarity_metrics_df = pd.DataFrame(index=range(n_simulations), columns= ['Schreiber', 'VanRossum', 'CorrMean', 'CorrStdv'])

for sim_id in range(n_simulations):

    print(f"Running Simulation: {sim_id}")

    # Jitter the Spike Train every loop (:
    simulated_spike_train = jitter_spike_train(spike_train_true[3,:], sample_rate, bin_size_ms = 50)

    # Compute the Metrics
    schreiber_val = schreiber_similarity_analysis(spike_train_true[3,:], simulated_spike_train, sample_rate, sigma_ms = 5)
    van_rossum_val = van_rossum_distance_analysis(spike_train_true[3,:], simulated_spike_train,  sample_rate, tau_ms = 10)
    correlation_results, *_ = run_multiscale_correlation_analysis(spike_train_true[3,:], simulated_spike_train,  sample_rate)

    # Insert the values
    similarity_metrics_df.loc[sim_id, 'Schreiber'] = float(schreiber_val)
    similarity_metrics_df.loc[sim_id, 'VanRossum'] = float(van_rossum_val)
    similarity_metrics_df.loc[sim_id, 'CorrMean']  = float(np.mean(correlation_results))
    similarity_metrics_df.loc[sim_id, 'CorrStdv']  = float(np.std(correlation_results))


# Create a basic histogram
plt.subplot(1,3,1)
schrieber_threshold = np.round(np.percentile(similarity_metrics_df['Schreiber'], 95),2)
sns.histplot(data=similarity_metrics_df['Schreiber'])
plt.xlabel('Schreiber Similarity')
plt.ylabel('Count')
plt.title(f"Schrieber 95th percentile: {schrieber_threshold}")

plt.subplot(1,3,2)
vanRossum_threshold = np.round(np.percentile(similarity_metrics_df['VanRossum'], 5),2)
sns.histplot(data=similarity_metrics_df['VanRossum'])
plt.xlabel('VanRossum Distance')
plt.ylabel('Count')
plt.title(f"VanRossum 5th percentile: {vanRossum_threshold}")

plt.subplot(1,3,3)
corr_threshold = np.round(np.percentile(similarity_metrics_df['CorrMean'], 95),2)
sns.histplot(data=similarity_metrics_df['CorrMean'])
plt.xlabel('Avg Correlation')
plt.ylabel('Count')
plt.title(f"Correlation 95th percentile: {corr_threshold}")
plt.show()