import torch
import torch.cuda
import os
import argparse
import dill
import numpy as np
from modules.gnn import graph_batch_from_smile, drug_sdf_db, drug_sdf_db_list
from util import buildPrjSmiles, buildMoleculeDDI, Regularization
from training import Test, Train, Pre_Train
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from modules import PreModel, BiMoRec
from modules import DTA
import random

def set_seed():
    torch.manual_seed(1203)
    np.random.seed(2048)

    random.seed(2048)
    torch.cuda.manual_seed(2048)
    torch.cuda.manual_seed_all(2048) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_model_name(args):
    model_name = [
        f'dim_{args.dim}',  f'lr_{args.lr}', f'lr_pretrain_{args.lr_pretrain}', f'coef_{args.coef}', f'regular_{args.regular}', f'gamma_{args.gamma}', f'layers_{args.layers}',
        f'dp_{args.dp}', f'ddi_{args.target_ddi}', f'epochs_pretrain_{args.epochs_pretrain}', f'target_ja_{args.target_ja}', f'dataset_{args.dataset}', f'model_{args.model}'
    ]
    return '-'.join(model_name)


def parse_args():
    parser = argparse.ArgumentParser('Experiment For DrugRec')
    parser.add_argument('--Test', action='store_true', help="evaluating mode")
    parser.add_argument('--dim', default=128, type=int, help='model dimension')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--lr_pretrain', default=2e-3, type=float, help='pretrain_learning rate')
    parser.add_argument('--dp', default=0.5, type=float, help='dropout ratio')
    parser.add_argument(
        '--model_name', type=str,
        help="the model name for training, if it's left blank,"
        " the code will generate a new model name on it own"
    )
    parser.add_argument(
        '--resume_path', type=str,
        help='path of well trained model, only for evaluating the model'
    )
    parser.add_argument(
        '--device', type=int, default=2,
        help='gpu id to run on, negative for cpu'
    )
    parser.add_argument(
        '--target_ddi', type=float, default=0.06,
        help='expected ddi for training'
    )
    parser.add_argument(
        '--target_ja', type=float, default=0.54,
        help='expected ja for training'
    )
    parser.add_argument(
        '--coef', default=2.5, type=float,
        help='coefficient for DDI Loss Weight Annealing'
    )
    parser.add_argument(
        '--epochs', default=20, type=int,
        help='the epochs for training'
    )
    parser.add_argument(
        '--epochs_pretrain', default=300, type=int,
        help='the epochs for training'
    )
    parser.add_argument(
        '--model', default='modelv3_test', type=str,
        help='code model of train'
    )
    parser.add_argument(
        '--only_pretrain', default=False, type=bool,
        help='only_pretrain'
    )
    parser.add_argument(
        '--only_train', default=True, type=bool,
        help='only_pretrain'
    )
    parser.add_argument("--layers", type=int, default=4, help="gin gvp layers number")
    parser.add_argument("--dataset", type=str, default='mimic3', help="dataset name")
    parser.add_argument("--gamma", type=float, default=0.5, help="scheduler parameter")
    parser.add_argument("--regular", type=float, default=0.008, help="regularization parameter")
    parser.add_argument("--resume_path_pretrained", default="../saved/pretrain_model_saved/pretrained_model_v2.pt", help="best pretrained model path")
    parser.add_argument("--use_best_pretrained", type=bool, default=True, help="use bestpretrained model")

    args = parser.parse_args()
    if args.Test and args.resume_path is None:
        raise FileNotFoundError('Can\'t Load Model Weight From Empty Dir')
    if args.model_name is None:
        args.model_name = get_model_name(args)

    return args


if __name__ == '__main__':
    set_seed()
    args = parse_args()
    print(args)
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    dataset_path = '../data/'+args.dataset+'/'

    data_path = dataset_path+'records_final.pkl'
    voc_path = dataset_path+'voc_final.pkl'
    ddi_adj_path = dataset_path+'ddi_A_final.pkl'
    ddi_mask_path = dataset_path+'ddi_mask_H.pkl'
    molecule_path = dataset_path+'idx2SMILES.pkl'
    substruct_smile_path = dataset_path+'substructure_smiles.pkl'
    subsmiles_3d_path = dataset_path+'substructure_smiles_3d.sdf'


    smiles2cid_path = '../data/SMILES2CID.pkl'
    smiles_3d_path = '../data/smiles_3d'

    with open(ddi_adj_path, 'rb') as Fin:
        ddi_A_final = dill.load(Fin)
        ddi_adj = torch.from_numpy(ddi_A_final).to(device)
        ddi_x = torch.from_numpy(np.random.rand(np.size(ddi_A_final, 0),1)).to(device)
        ddi_edge_index = ddi_adj.nonzero().t().contiguous()
    with open(ddi_mask_path, 'rb') as Fin:
        ddi_mask_H = torch.from_numpy(dill.load(Fin)).to(device)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
    with open(molecule_path, 'rb') as Fin:
        molecule = dill.load(Fin)
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)
    with open(smiles2cid_path, 'rb') as Fin:
        smiles2cid = dill.load(Fin)

    diag_voc, pro_voc, med_voc = \
        voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    )

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    average_projection, smiles_list = \
        buildPrjSmiles(molecule, med_voc.idx2word)
    average_projection = average_projection.to(device)

    molecule_ddi = buildMoleculeDDI(ddi_adj, ddi_mask_H)
    molecule_ddi = molecule_ddi.to(device)

    molecule_graphs = graph_batch_from_smile(smiles_list)
    molecule_forward = {'batched_data': molecule_graphs.to(device)}
    molecule_para = {
        'num_layer': args.layers, 'emb_dim': args.dim, 'graph_pooling': 'mean',
        'drop_ratio': args.dp, 'gnn_type': 'gin', 'virtual_node': False
    }

    with open(substruct_smile_path, 'rb') as Fin:
        substruct_smiles_list = dill.load(Fin)

    substruct_graphs = graph_batch_from_smile(substruct_smiles_list)
    substruct_forward = {'batched_data': substruct_graphs.to(device)}
    substruct_para = {
        'num_layer': args.layers, 'emb_dim': args.dim, 'graph_pooling': 'mean',
        'drop_ratio': args.dp, 'gnn_type': 'gin', 'virtual_node': False
    }

    ddi_para = {
        'num_layer': args.layers, 'emb_dim': args.dim,
        'drop_ratio': args.dp, 'gnn_type': 'gat', 'p_or_m': 'minus'
    }

    ddi_data = {
        'ddi_x': ddi_x, 'ddi_edge_index': ddi_edge_index
    }

    molecule_ddi_x = torch.from_numpy(np.random.rand(np.size(molecule_ddi, 0),1)).to(device)
    molecule_ddi_edge_index = molecule_ddi.nonzero().t().contiguous()
    molecule_ddi_data = {
        'ddi_x': molecule_ddi_x, 'ddi_edge_index': molecule_ddi_edge_index
    }

    # 3D mole
    drug_sdf_dict = drug_sdf_db(smiles_3d_path)
    drug_geo_list = [drug_sdf_dict[smiles2cid[x]].to(device) for x in smiles_list]

    # 3D submole
    drug_sub_list = [x.to(device) for x in drug_sdf_db_list(subsmiles_3d_path)]


    model = BiMoRec(global_para=molecule_para, substruct_para=substruct_para, ddi_para=ddi_para,
        emb_dim=args.dim, global_dim=args.dim, substruct_dim=args.dim,
        substruct_num=ddi_mask_H.shape[1], voc_size=voc_size, device=device,
        dropout=args.dp
    ).to(device)

    pre_model = PreModel(global_para=molecule_para, substruct_para=substruct_para, emb_dim=args.dim, num_layers=args.layers, device=device).to(device)

    regular = Regularization(model, args.regular, p=0)

    drug_data = {
        'substruct_data': substruct_forward,
        'mol_data': molecule_forward,
        'ddi_mask_H': ddi_mask_H,
        'tensor_ddi_adj': ddi_adj,
        'average_projection': average_projection,
        'ddi_data': ddi_data,
        'molecule_ddi_data': molecule_ddi_data,
        'molecule_ddi': molecule_ddi,
        'drug_geo_list': DTA(drug_geo_list),
        'drug_sub_list': DTA(drug_sub_list),
        'ddi_A_final': ddi_A_final
    }

    if args.Test:
        Test(model, args.resume_path, device, data_test, voc_size, drug_data)
        print('Test is done!')
    else:
        if not os.path.exists(os.path.join('../saved', args.model_name)):
            os.makedirs(os.path.join('../saved', args.model_name))
        log_dir = os.path.join('../saved', args.model_name)
        optimizer_pre = Adam(pre_model.parameters(), lr=args.lr_pretrain)
        scheduler_pre = StepLR(optimizer_pre, step_size=100, gamma=0.1)

        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
        if not args.only_pretrain and not args.only_train:
            
            pre_train_model = Pre_Train(pre_model, device, drug_data, optimizer_pre, scheduler_pre, log_dir, EPOCH=args.epochs_pretrain)
            Train(
                model, pre_train_model, device, data_train, data_eval, voc_size, drug_data,
                optimizer, scheduler, regular, log_dir, args.coef, args.target_ddi, args.target_ja, EPOCH=args.epochs
            )
        else:
            if args.only_pretrain and args.only_train:
                print('args is wrong!')
            else:
                if args.only_pretrain:
                    pre_train_model = Pre_Train(pre_model, device, drug_data, optimizer_pre, scheduler_pre, log_dir, EPOCH=args.epochs_pretrain)
                if args.only_train:
                    if args.use_best_pretrained:
                        pretrain_path = args.resume_path_pretrained
                    else:
                        pretrain_path = os.path.join('../saved', args.model_name, 'pretrained_model.pt')
                    with open(pretrain_path, 'rb') as Fin:
                         pre_model.load_state_dict(torch.load(Fin, map_location=device))
                    Train(
                        model, pre_model, device, data_train, data_eval, voc_size, drug_data,
                        optimizer, scheduler, regular, log_dir, args.coef, args.target_ddi, args.target_ja, EPOCH=args.epochs
                    )