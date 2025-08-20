import pandas as pd

"""
Connectivity Matrix:

Simple example on how to get users to provide their connectivity information.
This will allow us to better identify / trouble shoot issues
"""
# ------------------------------------------------------------------------------------------
# ------------------------------------ Set-up Example --------------------------------------
# Here I am setting up ideally how the user should set up their connectivity matrix
# The input neurons are columns and the neuorns recieving the input are the rows

# Create Simple Example of ground truth
data_truth = [[0, 0, 0, 0, -1],
              [0, 0, 0, -1, 0],
              [1, 1, 0, 0, 0],
              [1, 1, 0, 0, 0],
              [0, 0, -1, 0.8, 0]]

# Create a Simple Example of something connected incorrectly 
data_predicted = [[0, 0, 1, 0, 0],
                  [0, 0, 0, 1, -1],
                  [1, 0, 0, 0, 0],
                  [1, 0.5, 0, 0, 0],
                  [0, 0, -0.8, 1, 0]]

connectivity_true_df = pd.DataFrame(data_truth, 
    index=["Output NeuronA", "Output NeuronB", "Output NeuronC", "Output NeuronD", "OutputNeuronE"],
    columns=["Input NeuronA", "Input NeuronB", "Input NeuronC", "Input NeuronD", "Input NeuronE"]) # column labels

connectivity_predicted_df = pd.DataFrame(data_predicted, 
    index=["Output NeuronA", "Output NeuronB", "Output NeuronC", "Output NeuronD", "OutputNeuronE"],
    columns=["Input NeuronA", "Input NeuronB", "Input NeuronC", "Input NeuronD", "Input NeuronE"]) # column labels

# ---------------------------------------------------------------------------------------------
# ----------------------------Simple Function for ouput prompt---------------------------------
def identify_inconsistencies(logical_df, locations, prompt):
    for row_idx, col_idx in locations:
        row_label = logical_df.index[row_idx]
        col_label = logical_df.columns[col_idx]
        print( prompt + f" from {col_label} to {row_label}")


# ---------------------------------------------------------------------------------------------
# ------------------------------------ Checking Connections -----------------------------------
# Various Check-Points Below
# Aiden: Ideally this will be turned into a function where you submit the true connectivity and 
# predicted connectivity - then we can store all inconsistencies as prompts to send to Philip

# 1. First lets check what is connected

# 1a. Excitatory Connections Identified First
excitatory_logical = (connectivity_true_df > 0).astype(float) - (connectivity_predicted_df > 0).astype(float)

# PT 1: FIND ANY MISSING EXCITATORY CONNECTIONS
identify_labels = (excitatory_logical > 0)
locations = list(zip(*identify_labels.to_numpy().nonzero()))  # indices of True values
identify_inconsistencies(excitatory_logical, locations, "Missing Excitatory Connection")

# PT 2: FIND ANY EXTRA EXCITATORY CONNECTIONS
identify_labels = (excitatory_logical < 0)
locations = list(zip(*identify_labels.to_numpy().nonzero()))  # indices of True values
identify_inconsistencies(excitatory_logical, locations, "Extra Excitatory Connection")

# ---------------------------------------------------------------------------------------------
# 1b. Inhibitory Connections Identified First
inhibitory_logical = (connectivity_true_df < 0).astype(float) - (connectivity_predicted_df < 0).astype(float)

# PT 1: FIND ANY MISSING INHIBITORY CONNECTIONS
identify_labels = (inhibitory_logical > 0)
locations = list(zip(*identify_labels.to_numpy().nonzero()))  # indices of True values
identify_inconsistencies(inhibitory_logical, locations, "Missing Inhibitory Connection")

# PT 2: FIND ANY EXTRA INHIBITORY CONNECTIONS
identify_labels = (inhibitory_logical < 0)
locations = list(zip(*identify_labels.to_numpy().nonzero()))  # indices of True values
identify_inconsistencies(inhibitory_logical, locations, "Extra Inhibitory Connection")

# ---------------------------------------------------------------------------------------------
# --------------------------------- Checking Synaptic Strengths -------------------------------
# 2. Now lets check if their synapse strength is correct 

# Lets take the absolute so we can directly check if the actual weight is more or less
# Only compare the true weights 
logical_true = (connectivity_true_df != 0).astype(float)
synaptic_weight = (connectivity_true_df.abs() - connectivity_predicted_df.abs())*logical_true

# PT 1: IDENTIFY SYNAPTIC WEIGHTS THAT ARE NOT AS STRONG AS ORIGINAL
# if positive that means the true synaptic weight was stronger than the predicted resulting in positive #
identify_labels = (synaptic_weight > 0)
locations = list(zip(*identify_labels.to_numpy().nonzero()))  # indices of True values
identify_inconsistencies(synaptic_weight, locations, "Lower Synaptic Weight Connection")

# PT 2: IDENTIFY SYNAPTIC WEIGHTS THAT ARE STRONGER THAN ORIGINAL
# if negative that means the predicted synaptic weight was stronger than the true resulting in negative #
identify_labels = (synaptic_weight < 0)
locations = list(zip(*identify_labels.to_numpy().nonzero()))  # indices of True values
identify_inconsistencies(synaptic_weight, locations, "Higher Synaptic Weight Connection")
