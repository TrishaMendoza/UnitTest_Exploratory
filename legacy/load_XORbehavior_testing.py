import numpy as np
import pandas as pd
from create_XOR_data import *


import matplotlib.pyplot as plt
import seaborn as sns

# Set to Seaborn Style
sns.set()

# Variables for Generating Data
stimulus_run_time_ms = 3000
time_vector = np.linspace(1, stimulus_run_time_ms, stimulus_run_time_ms)

stimulus_a = [  300, 500, 700, 1200, 1500, 1700, 2100, 2500, 2700, 2900 ]
stimulus_b = [  100, 500, 900, 1200, 1500, 1900, 2300, 2500, 2700, 2800 ]

# Obtain the Unique Stimulus to loop through
spike_times = np.unique(np.array(stimulus_a + stimulus_b))

# ----------------------------------------------------------------------------------------------
# Setup the Ground Truth Model
spike_train_true = generate_ground_truth(stimulus_a.copy(), stimulus_b.copy(), stimulus_run_time_ms)

# Developing a Truth Table 
ground_truth = pd.DataFrame({
    'Input Neuron A' : [ 0, 1, 1],
    'Input Neuron B' : [ 1, 0, 1],

    'Sim_Count' : [0] * 3,
    'Output Neuron C' : [0] * 3,
    'Output Neuron D' : [0] * 3,
    'Output Neuron E' : [0] * 3 })


# Lets now Generate a Truth Table 
for time_val in spike_times:
    
    # Spike Times - Resulting Values 
    spike_results = spike_train_true[:, time_val: time_val + 20]

    # Extract the Neuron Values 
    input_A = np.sum(spike_results[0])
    input_B = np.sum(spike_results[1])
    output_C = np.sum(spike_results[2])
    output_D = np.sum(spike_results[3])
    output_E = np.sum(spike_results[4])

    # Match input combo
    for idx, row in ground_truth.iterrows():
        if row['Input Neuron A'] == input_A and row['Input Neuron B'] == input_B:
            ground_truth.loc[idx, 'Sim_Count'] += 1
            ground_truth.loc[idx, 'Output Neuron C'] += output_C
            ground_truth.loc[idx, 'Output Neuron D'] += output_D
            ground_truth.loc[idx, 'Output Neuron E'] += output_E
            break


# Now Find the Average 
ground_truth['Output Neuron C'] = ground_truth['Output Neuron C'] / ground_truth['Sim_Count']
ground_truth['Output Neuron D'] = ground_truth['Output Neuron D'] / ground_truth['Sim_Count']
ground_truth['Output Neuron E'] = ground_truth['Output Neuron E'] / ground_truth['Sim_Count']


# ---------------------------------------------------------------------------------------------
# Now run the Predicted

# Confusion Matrix
conf_matrix = pd.DataFrame({
    'TP' : [0],
    'TN' : [0],
    'FP' : [0],
    'FN' : [0] })

    # Developing a Truth Table 
predicted_table = pd.DataFrame({
    'Input Neuron A' : [ 0, 1, 1],
    'Input Neuron B' : [ 1, 0, 1],

    'Sim_Count' : [0] * 3,
    'Output Neuron C' : [0] * 3,
    'Output Neuron D' : [0] * 3,
    'Output Neuron E' : [0] * 3 })

# Obtain the spike train data
neuron_names = ['Output Neuron C', 'Output Neuron D', 'Output Neuron E']
spike_train_predicted = generate_predicted(stimulus_a, stimulus_b, stimulus_run_time_ms)

# CALCULATE THE CONFUSION TABLE - TRUE POSITIVES/ FALSE POSITIVES/ TRUE NEGATIVES/ FALSE NEGATIVES
for time_val in spike_times:

    # Spike Times - Resulting Values 
    spike_results = np.sum(spike_train_true[:, time_val: time_val + 20], axis = 1)

    for idx, row in ground_truth.iterrows():
        if row['Input Neuron A'] == spike_results[0] and row['Input Neuron B'] == spike_results[1]:
            predicted_table.loc[idx, 'Sim_Count'] += 1

            for neur_idx, neur_id in enumerate(neuron_names):
                predicted_table.loc[idx, neur_id] += spike_results[neur_idx]

                # Identify True Positives and False Negatives
                if ground_truth.loc[idx, neur_id] >= 1:
                    if spike_results[neur_idx] >= 1:
                        conf_matrix.loc[0,'TP'] += 1
                    else:
                        conf_matrix.loc[0,'FN'] += 1

                # Identify True Negative and False Positives
                else:
                    if spike_results[neur_idx] >= 1:
                        conf_matrix.loc[0,'FP'] += 1
                    else:
                        conf_matrix.loc[0,'TN'] += 1

print(predicted_table)

accuracy = (conf_matrix.loc[0,'TP'] + conf_matrix.loc[0,'TN']) / (conf_matrix.loc[0,'TP'] + conf_matrix.loc[0,'TN'] + conf_matrix.loc[0,'FN'] + conf_matrix.loc[0,'FP']) * 100
accuracy = np.round(accuracy, 2)
print( accuracy )

# Now Find the Average 
predicted_table['Output Neuron C'] = predicted_table['Output Neuron C'] / predicted_table['Sim_Count']
predicted_table['Output Neuron D'] = predicted_table['Output Neuron D'] / predicted_table['Sim_Count']
predicted_table['Output Neuron E'] = predicted_table['Output Neuron E'] / predicted_table['Sim_Count']
predicted_table.to_html("predicted_table.html", index=False)

heat_data = [
    [conf_matrix.loc[0, 'TP'], conf_matrix.loc[0, 'FN']],
    [conf_matrix.loc[0, 'FP'], conf_matrix.loc[0, 'TN']]
]
sns.heatmap(heat_data, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Pos', 'Pred Neg'], yticklabels=['True Pos', 'True Neg'])
plt.title(f'Confusion Matrix Model Accuracy: {accuracy} %')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
   

# ----------------------------------------------------------------------------------------------------
# Optional Plotting

plt.subplot(2,3,1)
sns.lineplot( x = time_vector, y = spike_train_true[0,:], label = 'Neuron_A')
sns.lineplot( x = time_vector, y = spike_train_predicted[0,:], label = 'Neuron_A_Pred')

plt.subplot(2,3,2)
sns.lineplot( x = time_vector, y = spike_train_true[1,:], label = 'Neuron_B')
sns.lineplot( x = time_vector, y = spike_train_predicted[1,:], label = 'Neuron_B_Pred')

plt.subplot(2,3,3)
sns.lineplot( x = time_vector, y = spike_train_true[2,:], label = 'Neuron_C')
sns.lineplot( x = time_vector, y = spike_train_predicted[2,:], label = 'Neuron_C_Pred')

plt.subplot(2,3,4)
sns.lineplot( x = time_vector, y = spike_train_true[3,:], label = 'Neuron_D')
sns.lineplot( x = time_vector, y = spike_train_predicted[3,:], label = 'Neuron_D_Pred')

plt.subplot(2,3,5)
sns.lineplot( x = time_vector, y = spike_train_true[4,:], label = 'Neuron_E')
sns.lineplot( x = time_vector, y = spike_train_predicted[4,:], label = 'Neuron_E_Pred')

plt.tight_layout()
plt.show()

