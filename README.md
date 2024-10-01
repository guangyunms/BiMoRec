# BiMoRec
This is the code repository for the paper `	Medication Recommendation via Dual Molecular Modalities and Multi-Step Enhancement`.

`data`: The data folder stores the dataset and processed files

- `smiles_3d`: Folder for storing 3D molecular data.
- `3d.py`: Test whether Smiles has corresponding 3D files and whether molecular substructures can generate 3D structures.
- `processing_4.py`: Processing the Mimic4 dataset.
- `substructure_smiles_3d.sdf`: Write all the 3D data of substructures in one SDF file.

`saved`: Pre-trained model parameter files and BiMoRec model parameter files can be used for testing.

`src`: Storage location of code. The overall structure of the code is based on some of MoleRec's code.

- `BiMoRec.py`: Main model file.
- `gnn`: Support files including GVP and GIN networks.
- `main.py`: Main program file, including parameter settings and pre data reading.
- `training.py`: Related training functions.