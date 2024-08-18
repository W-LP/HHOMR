import numpy as np
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import math

# MLP
def normalize_rows(matrix):
    # 计算每行的范数
    row_norms = np.linalg.norm(matrix, axis=1)
    # 将每行归一化
    normalized_matrix = matrix / row_norms[:, np.newaxis]
    return normalized_matrix
def drop_node(feats, drop_rate, training):
    n = feats.shape[0]
    drop_rates = th.FloatTensor(np.ones(n) * drop_rate)

    if training:

        masks = th.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats

    else:
        feats = feats * (1. - drop_rate)

    return feats


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.input_dropout = nn.Dropout(input_droprate)
        self.hidden_dropout = nn.Dropout(hidden_droprate)
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):

        if self.use_bn:
            x = self.bn1(x)
        x = self.input_dropout(x)
        x = F.relu(self.layer1(x))

        if self.use_bn:
            x = self.bn2(x)
        x = self.hidden_dropout(x)
        x = self.layer2(x)

        return x




class MMConv(nn.Module):
    def __init__(self, in_features, out_features,  moment=3, use_center_moment=False):
        super(MMConv, self).__init__()
        self.moment = moment
        self.use_center_moment = use_center_moment
        self.in_features = in_features

        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.w_att = Parameter(torch.FloatTensor(self.in_features * 2,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.w_att.data.uniform_(-stdv, stdv)
    def moment_calculation(self, x, adj_t, moment):
        mu = torch.spmm(adj_t, x)
        out_list = [mu]
        if moment > 1:
            if self.use_center_moment:# 使用中心矩时，计算二阶矩（方差）
                sigma = torch.spmm(adj_t, (x - mu).pow(2))
            else:# 不使用中心矩时，计算二阶矩（平方和）
                sigma = torch.spmm(adj_t, (x).pow(2))
            sigma[sigma == 0] = 1e-16   # 避免除零错误，将小于等于零的值设置为一个极小值
            sigma = sigma.sqrt()        # 对二阶矩取平方根，得到标准差
            out_list.append(sigma)      # 将标准差加入输出列表

            for order in range(3, moment+1):        #高阶矩
                gamma = torch.spmm(adj_t, x.pow(order))     # 计算阶数为 order 的矩
                # 处理负值情况
                mask_neg = None
                if torch.any(gamma == 0):
                    # 将等于零的值设置为一个极小值
                    gamma[gamma == 0] = 1e-16
                if torch.any(gamma < 0):
                    # 将小于零的值取相反数，并记录相应的掩码
                    mask_neg = gamma < 0
                    gamma[mask_neg] *= -1

                # 对阶数为 order 的矩取 1/order 次方根
                gamma = gamma.pow(1/order)
                if mask_neg != None:
                    gamma[mask_neg] *= -1
                # 将阶数为 order 的矩加入输出列表
                out_list.append(gamma)
        return out_list
    def attention_layer(self, moments, q):
            k_list = []
            # if self.use_norm:
            #     h_self = self.norm(h_self) # ln
            q = q.repeat(self.moment, 1) # N * m, D
            # output for each moment of 1st-neighbors
            k_list = moments
            attn_input = torch.cat([torch.cat(k_list, dim=0), q], dim=1)    #在第0维度拼接，然和与q在第1维度进行拼接
            attn_input = F.dropout(attn_input, 0.5, training=self.training)
            e = F.elu(torch.mm(attn_input, self.w_att)) # N*m, D
            attention = F.softmax(e.view(len(k_list), -1, self.out_features).transpose(0, 1), dim=1) # N, m, D  # 对注意力权重进行 softmax 归一化，得到注意力分布
            out = torch.stack(k_list, dim=1).mul(attention).sum(1) # N, D# 将每个矩按照注意力分布进行加权求和
            return out
    def forward(self, input, adj , h0 , lamda, alpha, l, beta=0.1):
        theta = math.log(lamda/l+1)
        h_agg = torch.spmm(adj, input)
        h_agg = (1-alpha)*h_agg+alpha*h0
        h_i = torch.mm(h_agg, self.weight)
        h_i = theta*h_i+(1-theta)*h_agg
        # h_moment = self.attention_layer(self.moment_calculation(input, adj, self.moment), h_i)
        h_moment = self.attention_layer(self.moment_calculation(h0, adj, self.moment), h_i)
        output = (1 - beta) * h_i + beta * h_moment
        return output

class HHOMR(nn.Module):
    def __init__(self, G, hid_dim, n_class, S, K, batchnorm, num_diseases, num_mirnas,
                 d_sim_dim, m_sim_dim, out_dim, dropout, slope,
                 nfeat, nlayers, nhidden, lamda, alpha, use_center_moment=False, moment=3,  # 添加的参数
                 node_dropout=0.5, input_droprate=0.0,
                 hidden_droprate=0.0):
        super(HHOMR, self).__init__()
        self.G = G
        self.hid_dim = hid_dim
        self.S = S
        self.K = K
        self.n_class = n_class
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], hid_dim, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], hid_dim, bias=False)
        self.m_fc1 = nn.Linear(out_dim + n_class, out_dim)
        self.d_fc1 = nn.Linear(out_dim + n_class, out_dim)
        # self.m_fc1 = nn.Linear(64, out_dim)
        # self.d_fc1 = nn.Linear(64, out_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.mlp = MLP(hid_dim, out_dim, n_class, input_droprate, hidden_droprate, batchnorm)

        self.dropout = node_dropout
        self.node_dropout = nn.Dropout(node_dropout)
        self.predict = nn.Linear(out_dim * 2, 1)
        ####################################################
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(MMConv(nhidden, nhidden, use_center_moment=use_center_moment, moment=moment))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        ####################################################
        self.linear1 = nn.Linear(2, 32)
        self.linear2 = nn.Linear(878, 32)
        self.down = nn.Linear(128, 64)
        ##################################################

    def forward(self, graph,  diseases, mirnas,node_type_feature1,node_type_feature2,Topo, training=True):

        self.G.apply_nodes(lambda nodes: {'z': self.dropout1(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.dropout1(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
        feats = self.G.ndata.pop('z')
        x = feats
        adj=self.G.adj().to_dense()


        if training:  # Training Mode
            feat0 = []

            # # 拓扑信息
            # node_type_feature1 = self.linear1(node_type_feature1)
            # node_type_feature2 = self.linear1(node_type_feature2)
            # Topo = self.linear2(Topo)
            # # 全图的节点拓扑feature
            # H1 = torch.cat((node_type_feature1, node_type_feature2), 0)
            # # cat上了节点类型的feature
            # H1 = torch.cat((H1, Topo), 1)
            # x = torch.cat((H1, x), 1)
            # x = self.down(x)

            x = torch.cat((Topo, x), 1)
            x = self.down(x)

            ######################################################
            _layers = []
            x = F.dropout(x, self.dropout, training=self.training)
            h = self.act_fn(self.fcs[0](x))
            _layers.append(h)
            for ind, conv in enumerate(self.convs):
                h = F.dropout(h, self.dropout, training=self.training)
                h = th.softmax(conv(h, adj, _layers[0], self.lamda, self.alpha, ind + 1), dim=-1)
            h = h.detach().numpy()
            h = normalize_rows(h)
            h = torch.tensor(h)

            h = F.dropout(h, self.dropout, training=self.training)



            ######################################################

            feat0 = th.log_softmax(self.mlp(h), dim=-1)

            #
            h_d = th.cat((feat0[:self.num_diseases], feats[:self.num_diseases]), dim=1)

            h_m = th.cat((feat0[self.num_diseases:], feats[self.num_diseases:]), dim=1)
            # h_d = feat0[:self.num_diseases]
            #
            # h_m = feat0[self.num_diseases:]

            h_m = self.dropout1(F.elu(self.m_fc1(h_m)))     # (495,64)
            h_d = self.dropout1(F.elu(self.d_fc1(h_d)))     # （383,64）
            # (878,64)
            h = th.cat((h_d, h_m), dim=0)

            # 这里的disease和mirnas就是顶点，其对应位置就顶点之间存在边的label：0或者1
            # 疾病顶点特征
            h_diseases = h[diseases]  # disease中有重复的疾病名称;(17376,64)
            # mirnas顶点的特征
            h_mirnas = h[mirnas]

            h_concat = th.cat((h_diseases, h_mirnas), 1)  # (17376,128)


            predict_score = th.sigmoid(self.predict(h_concat))
            return predict_score
        # else:  # Inference Mode
        #
        #     feat0 = th.log_softmax(self.mlp(feat0), dim=-1)
        #     h_d = th.cat((feat0[:self.num_diseases], feats[:self.num_diseases]), dim=1)
        #     h_m = th.cat((feat0[self.num_diseases:], feats[self.num_diseases:878]), dim=1)
        #
        #     h_m = self.dropout1(F.elu(self.m_fc1(h_m)))  # （383,64）
        #     h_d = self.dropout1(F.elu(self.d_fc1(h_d)))
        #     h = th.cat((h_d, h_m), dim=0)
        #
        #     h_diseases = h[diseases]  # disease中有重复的疾病名称;(17376,64)
        #     h_mirnas = h[mirnas]
        #
        #     h_concat = th.cat((h_diseases, h_mirnas), 1)  # (17376,128)
        #     predict_score = th.sigmoid(self.predict(h_concat))
        #     return predict_score


