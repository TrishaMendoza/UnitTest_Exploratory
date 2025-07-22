import numpy as np

def interspike_interval(spike_train, sampling_rate):

    # Create a time vector 
    time_vector = np.arange(1, len(spike_train))/ sampling_rate
    isi_dict = {}

    for ch in range(spike_train.shape[1]):

        # Identify timepoint where the spike happens
        indx = np.where(spike_train[:,ch] ==1)
        time_vals = time_vector[indx]

        isi_dict[ch] = np.diff(time_vals)

    return isi_dict



