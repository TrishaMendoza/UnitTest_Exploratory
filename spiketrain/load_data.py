import numpy as np 
import h5py

# General file path
file_path = "/Users/trishamendoza/Desktop/CarbonCopies/Code/GenerateData/IFNeuronData/"

# Load the ground truth data
with h5py.File(file_path + "bGround_Truth_data_10trials_100stims_180000ms.hdf5", "r") as f:

    # List the trial groups - list components of dict ('keys')
    trial = f['trial_0']

    # Extract Spike Train Data
    row_indices = [1, 3, 5, 7, 9]
    spike_train_true = np.transpose(trial['data'][:,row_indices])

# Load the predicted incorrect data
with h5py.File(file_path + "bNeuron_B_NotConnectedto_NeuronD_10trials_100stims_180000ms.hdf5", "r") as f:

    # List the trial groups - list components of dict ('keys')
    trial = f['trial_0']

    # Extract Spike Train Data
    row_indices = [1, 3, 5, 7, 9]
    spike_train_predicted = np.transpose(trial['data'][:,row_indices])




















# Optional - familiarization with data by printing table info
# columns = trial['data'].attrs['columns']
# print("Trials in the file:", list(f.keys()))
# print('columns labels:', columns)