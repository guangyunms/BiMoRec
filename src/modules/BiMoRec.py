from .gnn import SAB, SAB_ddi
from .gnn import GNNGraph, GNN_DDI, DrugGVPModel
import torch.nn.functional as F
from torch_geometric import loader
from torch.nn.modules.loss import _Loss
from torch import Tensor
import torch
import math

# 定义一个允许交互的注意力模块  
class InteractiveAttentionModule(torch.nn.Module):  
    def __init__(self, input_dim):  
        super(InteractiveAttentionModule, self).__init__()  
        self.scale = torch.sqrt(torch.tensor(input_dim, dtype=torch.float32))  # 缩放因子  
  
    def forward(self, repr1, repr2):  
        # 计算查询和键的点积，得到注意力分数  
        scores = torch.matmul(repr1, repr2.transpose(-1, -2)) / self.scale  
          
        # 应用 softmax 来获得归一化的注意力权重  
        attention_weights = F.softmax(scores, dim=-1)  
          
        # 使用注意力权重对值进行加权求和  
        weighted_repr2 = torch.matmul(attention_weights, repr2)  
          
        # 将加权后的 repr2 与原始的 repr1 相加或者进行其他形式的融合  
        output = repr1 + weighted_repr2  
          
        return output  

class Attention(torch.nn.Module):
    def __init__(self, hidden_dim, device=torch.device('cuda:0')):
        super(Attention, self).__init__()
        self.device = device
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, outputs):
        scores = self.linear(outputs)
        weights = F.softmax(scores, dim=1)
        weighted = torch.mul(outputs, weights)
        representations = weighted.sum(1)
        return representations

class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)

    def forward(self, main_feat, other_feat, fix_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)
        fix_feat = torch.diag(fix_feat)
        other_feat = torch.matmul(fix_feat, other_feat)
        O = torch.matmul(Attn, other_feat)

        return O

class NTXent(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''
    def __init__(self, norm: bool = True, tau: float = 0.1, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(NTXent, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / (torch.einsum('i,j->ij', z1_abs, z2_abs) + 1e-8)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        def cov_loss(x):
            batch_size, metric_dim = x.size()
            x = x - x.mean(dim=0)
            cov = (x.T @ x) / (batch_size - 1)
            off_diag_cov = cov.flatten()[:-1].view(metric_dim - 1, metric_dim + 1)[:, 1:].flatten()
            return off_diag_cov.pow_(2).sum() / metric_dim


        def std_loss(x):
            std = torch.sqrt(x.var(dim=0) + 1e-04)
            return torch.mean(torch.relu(1 - std))
        def uniformity_loss(x1: Tensor, x2: Tensor, t=2) -> Tensor:
            sq_pdist_x1 = torch.pdist(x1, p=2).pow(2)
            uniformity_x1 = sq_pdist_x1.mul(-t).exp().mean().log()
            sq_pdist_x2 = torch.pdist(x2, p=2).pow(2)
            uniformity_x2 = sq_pdist_x2.mul(-t).exp().mean().log()
            return (uniformity_x1 + uniformity_x2) / 2

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss

class PreModel(torch.nn.Module):
    def __init__(self, global_para, substruct_para, emb_dim, num_layers, device=torch.device('cuda:0'), *args, **kwargs):
        super(PreModel, self).__init__(*args, **kwargs)
        self.device = device

        self.drug_node_in_dim=[66, 1]
        self.drug_node_h_dims=[emb_dim, 64]
        self.drug_edge_in_dim=[16, 1]
        self.drug_edge_h_dims=[32, 1]
        self.drug_fc_dims=[1024, emb_dim]
        self.drug_emb_dim = self.drug_node_h_dims[0]

        self.global_encoder = GNNGraph(**global_para)

        self.substruct_encoder = GNNGraph(**substruct_para)

        self.drug_model = DrugGVPModel(
            node_in_dim=self.drug_node_in_dim, node_h_dim=self.drug_node_h_dims,
            edge_in_dim=self.drug_edge_in_dim, edge_h_dim=self.drug_edge_h_dims, num_layers=num_layers
        )

        self.drug_sub_model = DrugGVPModel(
            node_in_dim=self.drug_node_in_dim, node_h_dim=self.drug_node_h_dims,
            edge_in_dim=self.drug_edge_in_dim, edge_h_dim=self.drug_edge_h_dims, num_layers=num_layers
        )

        self.drug_fc = self.get_fc_layers(
            [self.drug_emb_dim] + self.drug_fc_dims,
            dropout=0.25, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
        
        self.drug_sub_fc = self.get_fc_layers(
            [self.drug_emb_dim] + self.drug_fc_dims,
            dropout=0.25, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
        

        self.ntx = NTXent()
    
    def get_fc_layers(self, hidden_sizes,
            dropout=0, batchnorm=False,
            no_last_dropout=True, no_last_activation=True):
        act_fn = torch.nn.LeakyReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(torch.nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(torch.nn.BatchNorm1d(out_dim))
        return torch.nn.Sequential(*layers)
    

    def forward(self, mol_data, drug_geo_list, substruct_data, drug_sub_list):

        global_embeddings = self.global_encoder(**mol_data)  # [284,64]
        drug_geo_loader = loader.DataLoader(dataset=drug_geo_list, batch_size=len(drug_geo_list), shuffle=False)
        xd = self.drug_model(next(iter(drug_geo_loader)))
        xd = self.drug_fc(xd) # [284,128]
        loss1 = self.ntx(global_embeddings, xd)

        sub_embeddings = self.substruct_encoder(**substruct_data)  # [284,64]
        drug_sub_loader = loader.DataLoader(dataset=drug_sub_list, batch_size=len(drug_sub_list), shuffle=False)
        xd2 = self.drug_sub_model(next(iter(drug_sub_loader)))
        xd2 = self.drug_sub_fc(xd2) # [284,128]
        loss2 = self.ntx(sub_embeddings, xd2)

        return loss1, loss2



class BiMoRec(torch.nn.Module):
    def __init__(
        self, global_para, substruct_para, ddi_para, emb_dim, voc_size,
        substruct_num, global_dim, substruct_dim, use_embedding=False,
        device=torch.device('cuda:0'), dropout=0.5, *args, **kwargs
    ):
        super(BiMoRec, self).__init__(*args, **kwargs)
        self.device = device
        self.use_embedding = use_embedding

        if self.use_embedding:
            self.substruct_emb = torch.nn.Parameter(
                torch.zeros(substruct_num, emb_dim)
            )
        else:
            self.substruct_encoder = GNNGraph(**substruct_para)

        self.global_encoder = GNNGraph(**global_para)

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim),
            torch.nn.Embedding(voc_size[2], emb_dim)
        ])

        self.seq_gru = torch.nn.GRU(voc_size[2], voc_size[2], batch_first=True)

        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()
        self.sab = SAB(substruct_dim, substruct_dim, 2, use_ln=True)
        self.sab_mol = SAB(global_dim, global_dim, 2, use_ln=True)
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(voc_size[2] * 2, voc_size[2])
        )

        self.repr_query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 3, emb_dim)
        )

        self.drug_query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, voc_size[2])
        )

        self.substruct_rela = torch.nn.Linear(emb_dim, substruct_num)

        # self.substruct_score = torch.nn.Linear(emb_dim, substruct_num)
        self.aggregator = AdjAttenAgger(
            global_dim, substruct_dim, max(global_dim, substruct_dim)
        )
        score_extractor = [
            torch.nn.Linear(substruct_dim, substruct_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(substruct_dim // 2, substruct_dim // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(substruct_dim // 4, 1)
        ]

        self.score_extractor = torch.nn.Sequential(*score_extractor)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        if self.use_embedding:
            torch.nn.init.xavier_uniform_(self.substruct_emb)

    def get_fc_layers(self, hidden_sizes,
            dropout=0, batchnorm=False,
            no_last_dropout=True, no_last_activation=True):
        act_fn = torch.nn.LeakyReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(torch.nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(torch.nn.BatchNorm1d(out_dim))
        return torch.nn.Sequential(*layers)


    def forward(
        self, substruct_data, mol_data, patient_data, ddi_data, molecule_ddi_data, molecule_ddi,
        ddi_mask_H, tensor_ddi_adj, average_projection, drug_geo_list, drug_sub_list, ddi_A_final
    ):
        score_seq = []


        adm_last = patient_data[-1][:]
        Idx1_last = torch.LongTensor([adm_last[0]]).to(self.device)
        Idx2_last = torch.LongTensor([adm_last[1]]).to(self.device)
        repr1_last = self.rnn_dropout(self.embeddings[0](Idx1_last))
        repr2_last = self.rnn_dropout(self.embeddings[1](Idx2_last))
        repr1_sum_last = torch.sum(repr1_last, keepdim=True, dim=1)
        repr2_sum_last = torch.sum(repr2_last, keepdim=True, dim=1)

        repr_flatten_last = torch.cat([repr1_sum_last.flatten(), repr2_sum_last.flatten()])
        repr_query_last = self.drug_query(repr_flatten_last)
        repr_weight_last = repr_query_last.unsqueeze(0)

        if len(patient_data) <= 1:
            score = repr_weight_last
        else:
            for adm_num, adm in enumerate(patient_data[:-1]):
                adm_new = adm[:]
                Idx1 = torch.LongTensor([adm_new[0]]).to(self.device)
                Idx2 = torch.LongTensor([adm_new[1]]).to(self.device)
                Idx3 = torch.LongTensor([adm_new[2]]).to(self.device)

                repr1 = self.rnn_dropout(self.embeddings[0](Idx1))
                repr2 = self.rnn_dropout(self.embeddings[1](Idx2))
                repr3 = self.rnn_dropout(self.embeddings[2](Idx3))

                repr1_sum = torch.sum(repr1, keepdim=True, dim=1)
                repr2_sum = torch.sum(repr2, keepdim=True, dim=1)
                repr3_sum = torch.sum(repr3, keepdim=True, dim=1)

                repr_flatten = torch.cat([repr1_sum.flatten(), repr2_sum.flatten(), repr3_sum.flatten()])

                repr_query = self.repr_query(repr_flatten)

                substruct_weight = torch.sigmoid(self.substruct_rela(repr_query))
                
                global_embeddings = self.global_encoder(**mol_data)
                selected_rows = average_projection[adm_new[2]]
                row_sums = torch.sum(selected_rows, axis=0)
                selected_embeddings = global_embeddings[row_sums > 0]
                enhanced_embeddings = self.sab_mol(selected_embeddings.unsqueeze(0))
                global_embeddings[row_sums > 0] = enhanced_embeddings.squeeze(0)

                global_embeddings = torch.mm(average_projection, global_embeddings)
                substruct_embeddings = self.sab(
                    self.substruct_emb.unsqueeze(0) if self.use_embedding else
                    self.substruct_encoder(**substruct_data).unsqueeze(0)
                ).squeeze(0)
                molecule_embeddings = self.aggregator(
                    global_embeddings, substruct_embeddings,
                    substruct_weight, mask=torch.logical_not(ddi_mask_H > 0)
                )
                diag_repr_weight_last = torch.diag(torch.sigmoid(repr_weight_last.squeeze(0)))
                molecule_embeddings = torch.matmul(diag_repr_weight_last, molecule_embeddings)
                score = self.score_extractor(molecule_embeddings).t().unsqueeze(0)
                score_seq.append(score)


            score_seq = torch.cat(score_seq, dim=1)
            output, hidden = self.seq_gru(score_seq)
            score_seq_repr = torch.cat([hidden], dim=-1)
            score_last_repr = torch.cat([output[:, -1]], dim=-1)
            drug_repr = torch.cat([score_seq_repr.flatten(), score_last_repr.flatten()])
            query = self.query(drug_repr)

            drug_weight = query.unsqueeze(0)
            score = drug_weight + repr_weight_last


        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)  # [131,131]
        batch_neg = 0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum()  # ddi score
        return score, batch_neg
