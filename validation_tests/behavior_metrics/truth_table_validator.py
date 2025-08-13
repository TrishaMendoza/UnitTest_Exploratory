import pandas as pd
import numpy as np

# -------------------------- Identify Behaviour Patterns --------------------------
def identify_behaviour_patterns(truth_table_df, input_spike_train, output_spike_train):

    """
    Identify Behaviour Patterns:
    The following code will identify input patterns in the spike train data to then further 
    identify if the ouput data behaves as it is expected too. The function will return relative 
    accuracies for each behaviour, overall accuracy for combined behaviors, and finally the
    TP/FP/TN/FN values for a confusion matrix 

    Input:
        1. truth_table_df = data frame containing both the expected binary inputs and binary outputs
        2. input_spike_train = the spike trains for all input neurons in the same order as outlined
                               in the truth table 
        3. output_spike_train = the spike trains for all output neurons in the same order as outlined 
                                in the truth table 
    
    Sample Truth Table Format:
    test_truth_table_df = pd.DataFrame({
        "Input NeuronA": [1,0,1],
        "Input NeuronB": [0,1,1],

        "Ouput NeuronC": [1,1,0] })
    
    Output:
    """

    # From Truth table extract input patterns + expected ouput pattern
    input_truth = truth_table_df.filter(like = 'Input').to_numpy()
    output_true = truth_table_df.filter(like = 'Ouput').to_numpy()

    # Ensure it is a 2D input for uniform calculations
    if input_spike_train.ndim == 1:
        input_spike_train = np.atleast_2d(input_spike_train)
    if output_spike_train.ndim == 1:
        output_spike_train = np.atleast_2d(output_spike_train)

    # Initialize Storage 
    initial_storage = []

    # Loop through each of the patterns
    for indx, rows in enumerate(input_truth):

        # Extract input + output one by one
        input_pattern = np.transpose(np.atleast_2d(input_truth[indx,:]))
        ouput_pattern = np.transpose(np.atleast_2d(output_true[indx,:]))

        # Match the input pattern to current spike train
        match = np.all(input_spike_train ==  input_pattern, axis=0)
        match_indices = np.where(match)[0]
        if match_indices.size != 0:

            # Send for statistical calculation
            TP, FN, TN, FP = calculate_performance( match_indices, ouput_pattern, output_spike_train)
            accuracy = (TN + TP) / (TN + FP + FN + TP)

            # Check if Values equate to zero to avoid error
            if TP + FN != 0:
                sensitivity = TP / (TP + FN)
            else:
                sensitivity = 0
            
            if TN + FP != 0:
                specificity = TN / (TN + FP)
            else:
                specificity = 0

            # Create a temporary dictionary to append
            initial_storage.append({'InputPattern': f"Pattern {indx+1}",
            'TP': TP, 'FN': FN, 'TN': TN, 'FP': FP, 
            'Accuracy': accuracy, 'Sensitivity': sensitivity, 'Specificity': specificity})
        else:

            # Create a temporary dictionary to append
            initial_storage.append({'InputPattern': f"Pattern {indx+1}",
            'TP': 0, 'FN': 0, 'TN': 0, 'FP': 0,
            'Accuracy': 0, 'Sensitivity': 0, 'Specificity': 0})
    
    # Finalize Storage Calculation
    performance_df = pd.DataFrame(initial_storage)

    # Calculate the total for the Numeric Columns
    totals = performance_df[['TP', 'FN', 'TN', 'FP']].sum()
    totals['InputPattern'] = 'Total'
    totals['Accuracy'] = (totals['TN'] + totals['TP']) / (totals['TN'] + totals['TP'] + totals['FP'] + totals['FN'])

    # Checking for potential errors
    if totals['TP'] + totals['FN'] != 0:
        totals['Sensitivity'] = totals['TP'] / (totals['TP'] + totals['FN'])
    else:
        totals['Sensitivity'] = np.nan

    if totals['TN'] + totals['FP'] != 0:
        totals['Specificity'] = totals['TN']  / (totals['TN'] + totals['FP'])
    else:
        totals['Specificity'] = np.nan

    performance_df = pd.concat([ performance_df, totals.to_frame().T], ignore_index=True)
    return performance_df



# -------------------------- Calculate the Statistics for Accuracy --------------------------
def calculate_performance(column_indices, true_pattern, predicted_spike_train):

    # Initiate the count
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Loop through the different indices and calculate TP and false positives
    for column_id in column_indices:

        # Get the predicted pattern output
        output_predicted = np.transpose(np.atleast_2d(predicted_spike_train[:, column_id]))

        # Lets first identify TP and FN
        if np.any(true_pattern == 1):
            one_indx = np.where(true_pattern == 1)[0]
            TP = TP + len(np.where(output_predicted[one_indx] == 1)[0])
            FN = FN + len(np.where(output_predicted[one_indx] == 0)[0])

        # Lets now identify TN and False Positives
        if np.any(true_pattern == 0):
            zero_indx = np.where(true_pattern == 0)[0]
            TN = TN + len(np.where(output_predicted[zero_indx] == 0)[0])
            FP = FP + len(np.where(output_predicted[zero_indx] == 1)[0])
    
    return TP, FN, TN, FP
 