
def structure_prompt(structure_df, locations, prompt):
    """
    Structure Prompt:
    this function will send in the matrix and identify the 
    location of the problem areas. The user will send in the corresponding 
    prompt fo this as well.

    Input
        1. structure_df: the original data frame indicating the input and output neurons
        2. locations: the locations of the problem indices
        3. prompt: the corresponding prompt indicating the issue 

    Output
        1. 
    """

    # Append all possible prompts
    prompts = []

    for row_idx, col_idx in locations:

        row_label = structure_df.index[row_idx]
        col_label = structure_df.columns[col_idx]
        temp_prompt = ( prompt + f" from {col_label} to {row_label}")
        prompts.append(temp_prompt)

    return prompts


def structure_analysis(structure_true_df, structure_predicted_df):
    """
    Structure Analysis:
    the following code will go through all possible scenarious of incorrect setup from
    extra excitatory/inhibitory connections, missing excitatory/inhibitory connections,
    and finally a difference in synaptic weights.

    Input:
        1. structure_true_df: connectivity matrix for true structure columns are the input neurons 
        and rows are the receiving/output neurons. Values should indicate their synaptic weight connection
        2. structure_predicted_df: connectivity matrix for predicted structure columns are the input neurons 
        and rows are the receiving/output neurons. Values should indicate their synaptic weight connection

    Output:
        1. finalized_prompts: list of all the problem prompts!

    """

    # Initialize the Prompt storage
    finalized_prompts = []

    # ---------------------------------------------------------------------------------------------
    # 1a. Excitatory Connections Identified First
    excitatory_logical = (structure_true_df > 0).astype(float) - (structure_predicted_df > 0).astype(float)

    # 1b. Develop Prompt for Missing Excitatory 
    identify_labels = (excitatory_logical > 0)
    locations = list(zip(*identify_labels.to_numpy().nonzero()))  
    prompt = structure_prompt(structure_true_df, locations, "Model is missing excitatory connection")
    finalized_prompts.append(prompt)

    # 1c. Develop Promp for Extra Excitatory
    identify_labels = (excitatory_logical < 0)
    locations = list(zip(*identify_labels.to_numpy().nonzero()))  
    prompt = structure_prompt(structure_true_df, locations, "Model contains extra excitatory connection")
    finalized_prompts.append(prompt)

    # ---------------------------------------------------------------------------------------------
    # 2a. Inhibitory Connections Identified First
    inhibitory_logical = (structure_true_df < 0).astype(float) - (structure_predicted_df < 0).astype(float)

    # 2b. Develop Prompt for Missing Inhibitory
    identify_labels = (inhibitory_logical > 0)
    locations = list(zip(*identify_labels.to_numpy().nonzero()))  
    prompt = structure_prompt(structure_true_df, locations, "Model is missing inhibitory connection")
    finalized_prompts.append(prompt)

    # 2c. Develop Promp for Extra Inhibitory
    identify_labels = (inhibitory_logical < 0)
    locations = list(zip(*identify_labels.to_numpy().nonzero()))  
    prompt = structure_prompt(structure_true_df, locations, "Model contains extra inhibitory connection")
    finalized_prompts.append(prompt)

    # ---------------------------------------------------------------------------------------------
    # 3a. From Correct Connection: Inspect Synaptic Weights
    logical_true = (structure_true_df != 0).astype(float)
    synaptic_weight = (structure_true_df.abs() - structure_predicted_df.abs())*logical_true

    # 3b. Identify Lower Synaptic Connections
    identify_labels = (synaptic_weight > 0)
    locations = list(zip(*identify_labels.to_numpy().nonzero()))  # indices of True values
    prompt = structure_prompt(structure_true_df, locations, "Lower synaptic weight than expected in connection")
    finalized_prompts.append(prompt)

    # 3c. Identify Higher Synaptic Connections
    identify_labels = (synaptic_weight < 0)
    locations = list(zip(*identify_labels.to_numpy().nonzero()))  # indices of True values
    prompt = structure_prompt(structure_true_df, locations, "Higher synaptic weight than expected in connection")
    finalized_prompts.append(prompt)

    return finalized_prompts



