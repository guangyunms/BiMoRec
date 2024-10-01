from ogb.utils import smiles2graph
from torch_geometric.data import Data
from pathlib import Path
import numpy as np
import torch
from .mol_graph import sdf_to_graphs, sdf_to_graphs_list

def graph_batch_from_smile(smiles_list):
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]
    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }
    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    return Data(**result)


def drug2sdf_file(drug_sdf_dir):
    drug2sdf_file = {f.stem : str(f) for f in Path(drug_sdf_dir).glob('*.sdf')}
    
    if all([k.isdigit() for k in drug2sdf_file.keys()]):
        drug2sdf_file = {int(k) : v for k, v in drug2sdf_file.items()}

    return drug2sdf_file

def drug_sdf_db(drug_sdf_dir):
    drug_files = drug2sdf_file(drug_sdf_dir)
    drug_sdf_db = sdf_to_graphs(drug_files)
    return drug_sdf_db

def drug_sdf_db_list(drug_sdf_path):
    drug_sdf_db_list = sdf_to_graphs_list(drug_sdf_path)
    return drug_sdf_db_list
