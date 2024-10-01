import torch.utils.data as data

class DTA(data.Dataset):
    """
    Base class for loading drug-target binding affinity datasets.
    Adapted from: https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py
    """
    def __init__(self, data_list=None):
        """
        Parameters
        ----------
            df : pd.DataFrame with columns [`drug`, `protein`, `y`],
                where `drug`: drug key, `protein`: protein key, `y`: binding affinity.
            data_list : list of dict (same order as df)
                if `onthefly` is True, data_list has the PDB coordinates and SMILES strings
                    {`drug`: SDF file path, `protein`: coordinates dict (`pdb_data` in `DTATask`), `y`: float}
                if `onthefly` is False, data_list has the cached torch_geometric graphs
                    {`drug`: `torch_geometric.data.Data`, `protein`: `torch_geometric.data.Data`, `y`: float}
                `protein` has attributes:
                    -x          alpha carbon coordinates, shape [n_nodes, 3]
                    -edge_index edge indices, shape [2, n_edges]
                    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
                    -name       name of the protein structure, string
                    -node_s     node scalar features, shape [n_nodes, 6]
                    -node_v     node vector features, shape [n_nodes, 3, 3]
                    -edge_s     edge scalar features, shape [n_edges, 39]
                    -edge_v     edge scalar features, shape [n_edges, 1, 3]
                    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
                    -seq_emb    sequence embedding (ESM1b), shape [n_nodes, 1280]
                `drug` has attributes:
                    -x          atom coordinates, shape [n_nodes, 3]
                    -edge_index edge indices, shape [2, n_edges]
                    -node_s     node scalar features, shape [n_nodes, 66]
                    -node_v     node vector features, shape [n_nodes, 1, 3]
                    -edge_s     edge scalar features, shape [n_edges, 16]
                    -edge_v     edge scalar features, shape [n_edges, 1, 3]
                    -name       name of the drug, string
            onthefly : bool
                whether to featurize data on the fly or pre-compute
            prot_featurize_fn : function
                function to featurize a protein.
            drug_featurize_fn : function
                function to featurize a drug.
        """
        super(DTA, self).__init__()

        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        drug = self.data_list[idx]
        return drug

