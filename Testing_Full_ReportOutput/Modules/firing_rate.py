import numpy as np

def firing_rate(spike_train, window_ms, overlap, sampling_rate ):
    """
    Calculate firing rate using a sliding window.

    Parameters:
    - spike_train: binary array (1s = spike, 0s = no spike)
    - window_ms: window size in milliseconds
    - overlap: overlap between windows (percentage: 0–100)
    - fs: sampling rate in Hz

    Returns:
    - firing_rates: array of firing rates (Hz)
    - fano factor
    """

    window_size = int((window_ms / 1000) * sampling_rate)
    step_size = int(window_size * (1 - overlap / 100))

    firing_rates = []

    for start in range(0, len(spike_train) - window_size + 1, step_size):
        end = start + window_size
        window = spike_train[start:end, :]
        spike_count = np.sum(window, axis = 0)
        rate = spike_count / (window_size / sampling_rate)  # spikes per second (Hz)

        firing_rates.append(rate)


    return np.array(firing_rates)
