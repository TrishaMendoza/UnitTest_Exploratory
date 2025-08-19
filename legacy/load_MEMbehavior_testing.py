import numpy as np
import pandas as pd
from create_Recurrent_data import *

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('pastel') 

def generate_stimulus_times_linspace(start_time_ms, window_duration_ms, firing_rate_hz):
    n_spikes = int(window_duration_ms * firing_rate_hz / 1000)
    if n_spikes < 1:
        return []
    spike_times = np.linspace(start_time_ms, start_time_ms + window_duration_ms, n_spikes, endpoint=False)
    return spike_times.astype(int).tolist()

# Set to Seaborn Style
sns.set()

# Variables for Generating Data
stimulus_run_time_ms = 3000
time_vector = np.linspace(0, stimulus_run_time_ms, stimulus_run_time_ms) 

# Generate a Loop with Different Firing Frequencies 
stim_duration = 1000
rate_tests = [1, 5, 10, 20, 50, 75, 100, 150, 200]

true_rate = np.zeros([2, len(rate_tests)])
pred_rate = np.zeros([2, len(rate_tests)])


for indx, rate_val in enumerate(rate_tests):

    # Get the Testing rates
    stimulus_times = generate_stimulus_times_linspace(500, stim_duration, rate_val)

    # Extract the Ground Truth + Predicted
    spike_train_true = generate_recurrent_ground_truth(stimulus_times.copy(), stimulus_run_time_ms)
    spike_train_pred = generate_recurrent_predicted(stimulus_times.copy(), stimulus_run_time_ms)

    temp_true = np.sum(spike_train_true, axis=1)
    temp_pred = np.sum(spike_train_pred, axis=1)

    true_rate[:, indx] = temp_true[-2:]
    pred_rate[:, indx] = temp_pred[-2:]

plt.subplot(1,2,1)
rmse_b = np.sqrt(np.mean(np.square(true_rate[0,:] - pred_rate[0,:])))
sns.lineplot(x = rate_tests, y = true_rate[0,:], label = 'True', color='#A3D5FF')
sns.lineplot(x = rate_tests, y = pred_rate[0,:], label = 'Pred', color='#FFB3BA')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title(f'RMSE Neuron B: {np.round(rmse_b)}')

plt.subplot(1,2,2)
rmse_c = np.sqrt(np.mean(np.square(true_rate[1,:] - pred_rate[1,:])))
sns.lineplot(x = rate_tests, y = true_rate[1,:], label = 'True', color='#A3D5FF')
sns.lineplot(x = rate_tests, y = pred_rate[1,:], label = 'Pred', color='#FFB3BA')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title(f'RMSE Neuron C: {np.round(rmse_c)}')
plt.show()





# Plot the spike trains just to ensure the
plt.subplot(3,1,1)
sns.lineplot(x = time_vector , y = spike_train_true[0], label= 'Input Neuron A')
sns.lineplot(x = time_vector , y = spike_train_true[1], label= 'Ouput Neuron B')

plt.subplot(3,1,2)
sns.lineplot(x = time_vector , y = spike_train_true[0], label= 'Input Neuron A')
sns.lineplot(x = time_vector , y = spike_train_true[2], label= 'Ouput Neuron C')

plt.subplot(3,1,3)
sns.lineplot(x = time_vector , y = spike_train_true[1], label= 'Ouput Neuron B')
sns.lineplot(x = time_vector , y = spike_train_true[2], label= 'Output Neuron C')
plt.show()