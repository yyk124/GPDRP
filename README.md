# GPDRP
Requirement:
Torch1110，Python3.7, Torch_geometric, Rdkit

Note:
Four models (GIN_TRANSFORMER, GIN, GAT, GCN) are trained separately.


Information about the "data.zip" file:
cell_ge.csv: cell line gene pathway activity scores.
drug_cl_ic.csv: 80056 drug-cell line pair and their response LNIC50.
drugsmile_GDSC: 173 drug Canonical SMILES. 

Information about preprocess.py: create data in pytorch format.

Information about training.py: train a GPDRP model.

This returns the model and result files for the modelling achieving the best MSE for testing data throughout the training.

Information about utils.py: include TestbedDataset used by preprocess.py to create data, performance measures and functions to draw loss, pearson by epoch.

Information about the "GIN_TRANSFORMER.zip" file, "GIN.zip" file, "GAT.zip" file, "GCN.zip" file:
include model and results file.
