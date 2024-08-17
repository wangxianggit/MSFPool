from torch_geometric.nn import DenseGCNConv
import torch.nn as nn
from node_modules import graph_transformer,Dense_GIN,Dense_GIN_layer
import torch
class GCN(nn.Module):
    def __init__(self, conf):
        super(GCN,self).__init__()

        self.num_layers = conf.num_layers
        self.gcn = nn.ModuleList()
        if self.num_layers == 1:
            self.gcn.append(DenseGCNConv(conf.num_features, conf.hidden_dim))
        else:
            self.gcn.append(DenseGCNConv(conf.num_features, conf.hidden_dim))
            for i in range(1, self.num_layers):
                self.gcn.append(DenseGCNConv(conf.hidden_dim, conf.hidden_dim))
            # self.gcn.append(DenseGCNConv(hidden_dim, output_dim))

        self.mlp = nn.Sequential(
            nn.Linear(conf.hidden_dim, conf.hidden_dim),
            nn.ReLU(),
            nn.Linear(conf.hidden_dim, conf.num_task),
            # nn.ReLU()
        )

    def forward(self, data):
        x = data.x
        adj = data.adj
        for i in range(self.num_layers):
            x = self.gcn[i](x, adj,).view(x.shape[0], -1)
        out = self.mlp(x)

        return out


class LT(nn.Module):
    def __init__(self, conf):
        super(LT, self).__init__()
        self.embeding_method = Dense_GIN(3, conf.num_features, conf.hidden_dim)
        self.trans = graph_transformer(conf)
        # self.conv = nn.ModuleList()
        # for i in range(2):
        #     gin_mlp = nn.Linear(in_features=conf.hidden_dim, out_features=conf.hidden_dim, bias=True)
        #     self.conv.append(Dense_GIN_layer(nn=gin_mlp))
        # self.aggration = nn.LSTM((conf.hidden_dim * 2),
        #                          conf.hidden_dim, bidirectional=True, batch_first=True)
        self.mlp = nn.Linear(conf.hidden_dim, conf.num_task)
        self.action_method = nn.ReLU()
        # self.alpha = nn.Parameter(torch.Tensor([0.5]))
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        """节点嵌入"""
        out = self.embeding_method(data.x, data.adj).reshape(data.x.shape[0], -1)
        out = self.dropout(out)
        # feature_conv = out
        # feature_conv_li = []
        # for i in range(2):
        #     feature_conv = self.conv[i](feature_conv,data.adj).reshape(data.x.shape[0], -1)
        #     feature_conv = self.action_method(feature_conv)
        #     feature_conv_li.append(feature_conv)
        # feature_conv = torch.stack(feature_conv_li, dim=-1).view(data.x.shape[0], -1)

        # att, hc = self.aggration(feature_conv)

        # h, c = hc[0].transpose(0, 1), hc[1].transpose(0, 1)

        # feature_conv = torch.cat([h, c], dim=1).sum(dim=1)

        feature_trans = self.trans(out, data.adj)

        # out = feature_trans * self.alpha + feature_conv * (1-self.alpha)
        out = self.dropout(feature_trans)
        out = self.mlp(out)


        return out


