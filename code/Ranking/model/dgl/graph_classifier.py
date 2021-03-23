from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.relation_list = list(self.relation2id.values())
        self.no_jk = self.params.no_jk
        self.neg_list = None
        self.valid = False
        self.link_mode = 6
        self.is_big_dataset = False
        if self.params.dataset in ['fb237_v4', 'nell_v4', 'fb_new', 'WN_new']:
            self.is_big_dataset = True

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        torch.nn.init.normal_(self.rel_emb.weight)

        self.rel_depen = nn.ModuleList([nn.Embedding(self.params.num_rels, self.params.num_rels) for _ in range(self.link_mode)])

        for i in range(self.link_mode):
            if self.params.dataset in ['fb237_v1', 'nell_v4']:  # added 1
                torch.nn.init.uniform_(self.rel_depen[i].weight)
            else:
                torch.nn.init.normal_(self.rel_depen[i].weight)

        if params.six_mode:
            self.fc_reld = nn.ModuleList([nn.Linear(self.params.rel_emb_dim, self.params.rel_emb_dim)
                                          for _ in range(6)])
        else:
            self.fc_reld = nn.ModuleList([nn.Linear(self.params.rel_emb_dim, self.params.rel_emb_dim)
                                          for _ in range(self.link_mode)])
            if self.params.dataset in ['nell_v4']:
                for i in range(self.link_mode):
                    nn.init.kaiming_uniform_(self.fc_reld[i].weight, nonlinearity='relu')  # init 5

        self.conc = nn.Linear(self.params.rel_emb_dim * 2, self.params.rel_emb_dim)
        # if self.params.dataset in ['nell_v4']:
        #     nn.init.kaiming_normal_(self.conc.weight, nonlinearity='relu')  # worse for fb
        # else:
        #     nn.init.xavier_normal_(self.conc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.conc.weight, gain=nn.init.calculate_gain('relu'))

        num_final_gcn = self.params.num_gcn_layers
        if self.no_jk:
            num_final_gcn = 1

        if self.params.add_ht_emb:
            if self.params.ablation == 0:
                self.fc_layer = nn.Linear(3 * num_final_gcn * self.params.emb_dim + self.params.rel_emb_dim, 1)
            elif self.params.ablation == 1:  # no-subg
                self.fc_layer = nn.Linear(2 * num_final_gcn * self.params.emb_dim + self.params.rel_emb_dim, 1)
            elif self.params.ablation == 2:  # no-ent
                self.fc_layer = nn.Linear(1 * num_final_gcn * self.params.emb_dim + self.params.rel_emb_dim, 1)
            elif self.params.ablation == 3:  # only-rel
                self.fc_layer = nn.Linear(self.params.rel_emb_dim, 1)

        else:
            self.fc_layer = nn.Linear(num_final_gcn * self.params.emb_dim + self.params.rel_emb_dim, 1)

    def forward(self, data):
        if self.params.gpu >= 0:
            device = torch.device('cuda:%d' % self.params.gpu)
        else:
            device = torch.device('cpu')

        g, rel_labels = data

        local_g = g.local_var()
        in_deg = local_g.in_degrees(range(local_g.number_of_nodes())).float().numpy()
        in_deg[in_deg == 0] = 1
        node_norm = 1.0 / in_deg
        local_g.ndata['norm'] = node_norm
        local_g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
        norm = local_g.edata['norm']

        if self.params.gpu >= 0:
            norm = norm.cuda(device=self.params.gpu)
        g.ndata['h'] = self.gnn(g, norm)

        g_out = mean_nodes(g, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        u_node, v = head_ids, tail_ids
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()
        u_in_edge = g.in_edges(u_node, 'all')
        u_out_edge = g.out_edges(u_node, 'all')
        v_in_edge = g.in_edges(v, 'all')
        v_out_edge = g.out_edges(v, 'all')

        in_edge_out = torch.sparse_coo_tensor(torch.cat((u_in_edge[1].unsqueeze(0), u_in_edge[2].unsqueeze(0)), 0),
                                              torch.ones(len(u_in_edge[2])), size=torch.Size((num_nodes, num_edges)))
        out_edge_out = torch.sparse_coo_tensor(torch.cat((u_out_edge[0].unsqueeze(0), u_out_edge[2].unsqueeze(0)), 0),
                                               torch.ones(len(u_out_edge[2])), size=torch.Size((num_nodes, num_edges)))
        in_edge_in = torch.sparse_coo_tensor(torch.cat((v_in_edge[1].unsqueeze(0), v_in_edge[2].unsqueeze(0)), 0),
                                             torch.ones(len(v_in_edge[2])), size=torch.Size((num_nodes, num_edges)))
        out_edge_in = torch.sparse_coo_tensor(torch.cat((v_out_edge[0].unsqueeze(0), v_out_edge[2].unsqueeze(0)), 0),
                                              torch.ones(len(v_out_edge[2])), size=torch.Size((num_nodes, num_edges)))

        if self.is_big_dataset:  # smaller memory
            in_edge_out = self.sparse_index_select(in_edge_out, u_node).to(device=device)
            out_edge_out = self.sparse_index_select(out_edge_out, u_node).to(device=device)
            in_edge_in = self.sparse_index_select(in_edge_in, v).to(device=device)
            out_edge_in = self.sparse_index_select(out_edge_in, v).to(device=device)
        else:  # faster calculation
            in_edge_out = in_edge_out.to(device=device).to_dense()[u_node].to_sparse()
            out_edge_out = out_edge_out.to(device=device).to_dense()[u_node].to_sparse()
            in_edge_in = in_edge_in.to(device=device).to_dense()[v].to_sparse()
            out_edge_in = out_edge_in.to(device=device).to_dense()[v].to_sparse()

        ## six patterns
        edge_mode_5 = out_edge_out.mul(in_edge_in) # multi-edge
        edge_mode_6 = in_edge_out.mul(out_edge_in) # inverse relation
        out_edge_out = out_edge_out.sub(edge_mode_5)
        in_edge_in = in_edge_in.sub(edge_mode_5)
        in_edge_out = in_edge_out.sub(edge_mode_6)
        out_edge_in = out_edge_in.sub(edge_mode_6)
        # step by step
        # edge_connect_l = [in_edge_out, out_edge_out, in_edge_in, out_edge_in]
        edge_connect_l = [in_edge_out, out_edge_out, in_edge_in, out_edge_in, edge_mode_5, edge_mode_6]

        norm_sparse = []
        for i in range(self.link_mode):
            norm_sparse.append(torch.sparse.mm(edge_connect_l[i], torch.ones(num_edges, 1).to(device=device)) + 1e-30)

        if self.params.epoch <= (self.params.num_epochs + 1)//2:

            rel_neighbor_embd = sum([torch.sparse.mm(edge_connect_l[i],
                self.fc_reld[i](self.rel_emb(g.edata['type']))) * 1. / norm_sparse[i] for i in range(self.link_mode)]) * 1. / self.link_mode

        else:
            if self.rel_emb.weight.requires_grad:
                self.rel_emb.weight.requires_grad = False

            rel_neighbor_embd = sum([torch.sparse.mm(
                self.sparse_dense_mul(edge_connect_l[i],
                                      torch.softmax(self.rel_depen[i].weight, dim=1)[rel_labels][:, g.edata['type']]),
                self.fc_reld[i](self.rel_emb(g.edata['type']))) * 1. / norm_sparse[i] for i in range(self.link_mode)]) * 1. / self.link_mode

        rel_final_emb = self.conc(torch.cat([rel_neighbor_embd, self.rel_emb(rel_labels)], dim=-1))
        rel_final_emb = F.relu(rel_final_emb)
        rel_final_emb = F.normalize(rel_final_emb, p=2, dim=-1)

        neg_relations = []
        if self.valid:
            num_neg_rel = len(self.relation_list) - 1
        else:
            num_neg_rel = min(self.params.num_neg_samples_per_link, len(self.relation_list) - 1)
        for idx, itm in enumerate(rel_labels):
            rel_list = self.relation_list.copy()
            rel_list.remove(itm)
            neg_rel_list = np.random.choice(rel_list, num_neg_rel, replace=False)
            neg_relations.append(neg_rel_list)
        self.neg_list = neg_relations[-1]
        neg_relations = torch.tensor(neg_relations)
        if self.params.gpu >= 0:
            neg_relations = neg_relations.cuda(device=self.params.gpu)

        rel_neg_emb = self.conc(torch.cat(
            [rel_neighbor_embd.unsqueeze(1).expand_as(self.rel_emb(neg_relations)), self.rel_emb(neg_relations)],
            dim=-1))
        rel_neg_emb = F.relu(rel_neg_emb)
        rel_neg_emb = F.normalize(rel_neg_emb, p=2, dim=-1)

        if self.no_jk:
            if self.params.ablation == 0:
                g_rep = torch.cat([g_out.view(-1, self.params.emb_dim),
                                   head_embs.view(-1, self.params.emb_dim),
                                   tail_embs.view(-1, self.params.emb_dim),
                                   rel_final_emb], dim=1)
                g_rep_neg = torch.cat([g_out.view(-1, self.params.emb_dim).unsqueeze(1).repeat(1, num_neg_rel, 1),
                                       head_embs.view(-1, self.params.emb_dim).unsqueeze(1).repeat(1, num_neg_rel, 1),
                                       tail_embs.view(-1, self.params.emb_dim).unsqueeze(1).repeat(1, num_neg_rel, 1),
                                       rel_neg_emb], dim=2)
            elif self.params.ablation == 1:  # no-subg
                g_rep = torch.cat([head_embs.view(-1, self.params.emb_dim),
                                   tail_embs.view(-1, self.params.emb_dim),
                                   rel_final_emb], dim=1)
                g_rep_neg = torch.cat([head_embs.view(-1, self.params.emb_dim).unsqueeze(1).repeat(1, num_neg_rel, 1),
                                       tail_embs.view(-1, self.params.emb_dim).unsqueeze(1).repeat(1, num_neg_rel, 1),
                                       rel_neg_emb], dim=2)
            elif self.params.ablation == 2:  # no-ent
                g_rep = torch.cat([g_out.view(-1, self.params.emb_dim),
                                   rel_final_emb], dim=1)
                g_rep_neg = torch.cat([g_out.view(-1, self.params.emb_dim).unsqueeze(1).repeat(1, num_neg_rel, 1),
                                       rel_neg_emb], dim=2)
            elif self.params.ablation == 3:  # only-rel
                g_rep = torch.cat([rel_final_emb], dim=1)
                g_rep_neg = torch.cat([rel_neg_emb], dim=2)
        else:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               rel_final_emb], dim=1)

            g_rep_neg = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim).unsqueeze(1).repeat(
                1, num_neg_rel, 1),
                head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim).unsqueeze(
                    1).repeat(1, num_neg_rel, 1),
                tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim).unsqueeze(
                    1).repeat(1, num_neg_rel, 1),
                rel_neg_emb], dim=2)

        output = self.fc_layer(g_rep)
        output_neg = self.fc_layer(g_rep_neg)
        return output, output_neg

    @staticmethod
    def sparse_dense_mul(s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    @staticmethod
    def sparse_index_select(s, idx):
        indices_s = s._indices()
        indice_new_1 = torch.tensor([])
        indice_new_2 = torch.tensor([])
        num_i = 0.0
        for itm in idx:
            mask = (indices_s[0] == itm)
            indice_tmp_1 = torch.ones(sum(mask)) * num_i
            indice_tmp_2 = indices_s[1][mask].float()
            indice_new_1 = torch.cat((indice_new_1, indice_tmp_1), dim=0)
            indice_new_2 = torch.cat((indice_new_2, indice_tmp_2), dim=0)
            num_i = num_i + 1.0
        indices_new = torch.cat((indice_new_1.unsqueeze(0), indice_new_2.unsqueeze(0)), dim=0).long()

        return torch.sparse.FloatTensor(indices_new, torch.ones(indices_new.shape[1]),
                                        torch.Size((len(idx), s.shape[1])))
