import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
import math

class GCN_Dense(nn.Module):

    def __init__(self,input_dim,output_dim=0,hidden_dim=0,num_layers=1):
        super(GCN_Dense,self).__init__()
        self.num_layers = num_layers

        if num_layers == 1:
            self.GCN = Dense_GCN_layer(in_channels=input_dim,out_channels=output_dim)

        else:

            self.GCN = nn.ModuleList([])
            self.GCN.append(Dense_GCN_layer(in_channels=input_dim,out_channels=hidden_dim))

            for i in range(1, num_layers):

                self.GCN.append(Dense_GCN_layer(in_channels=hidden_dim,out_channels=output_dim))

    def forward(self,x,adj,mask):

        out = self.GCN[0](x,adj,mask)
        if self.num_layers>1:
            for i in range(1,self.num_layers):
                out = self.GCN[i](out,adj,mask)

        return out


class Dense_GCN_layer(nn.Module):

    def __init__(self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        bias: bool = True,):
        super(Dense_GCN_layer,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)


    def forward(self, x: torch.Tensor, adj:  torch.Tensor, mask = None,
                add_loop: bool = True) ->  torch.Tensor:

        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = self.lin(x)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


class Dense_GIN(nn.Module):

    def __init__(self,num_layer,input_dim,output_dim,hidden_dim=0,eps: float = 0.0,train_eps: bool = False,):
        super(Dense_GIN, self).__init__()

        self.num_layer = num_layer

        self.model = nn.ModuleList()

        if self.num_layer == 1:
            emdedimg_mlp = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=output_dim),
                nn.ReLU6()
            )
            self.model.append(Dense_GIN_layer(emdedimg_mlp))
        elif self.num_layer == 2:
            emdedimg_mlp = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_dim,bias=True),
                nn.ReLU6()
            )
            out_mlp =  nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=output_dim,bias=True),
                nn.ReLU6()
            )
            self.model.append(Dense_GIN_layer(emdedimg_mlp))
            self.model.append(Dense_GIN_layer(out_mlp))
        else:
            emdedimg_mlp = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_dim,bias=True),
                nn.ReLU6()
            )
            hidden_mlp = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim,bias=True),
                nn.ReLU6()
            )
            out_mlp = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=output_dim,bias=True),
                nn.ReLU6()
            )
            self.model.append(Dense_GIN_layer(emdedimg_mlp))
            for i in range(1,self.num_layer-1):
                self.model.append(Dense_GIN_layer(hidden_mlp))
            self.model.append(Dense_GIN_layer(out_mlp))


    def forward(self,x,adj,mask):

        for i in range(self.num_layer):
            x = self.model[i](x,adj,mask)

        return x





class Dense_GIN_layer(nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GINConv`."""
    def __init__(self,nn,eps: float = 0.0, train_eps: bool = False,
    ):
        super(Dense_GIN_layer,self).__init__()

        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.eps.data.fill_(self.initial_eps)



    def forward(self, x, adj, mask = None,
                add_loop: bool = True):

        B, N, _ = adj.size()

        out = torch.matmul(adj, x)
        if add_loop:
            out = (1 + self.eps) * x + out

        out = self.nn(out)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


class attention_layer(nn.Module):

    def __init__(self,dim,num_heads,lpapas_ratio,qkv_bias):
        super(attention_layer,self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        if lpapas_ratio == -1 :
            self.lpapas_ratio = nn.Parameter(torch.Tensor([0.5]))
        else:
            self.lpapas_ratio = lpapas_ratio

        head_dim = self.dim // self.num_heads

        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=self.qkv_bias)

        self.attn_drop = nn.Dropout(0.3)
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     self.qkv.reset_parameters()

    def forward(self,x,adj,mask):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn * mask.view(B,1, 1, N).to(x.dtype)

        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        lpapas = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2) * mask.view(B, N, 1).to(x.dtype)



        attn = attn.softmax(dim=-1) * (1 - self.lpapas_ratio) + self.lpapas_ratio * lpapas.view(B,1,N,N).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return x * mask.view(B, N, 1).to(x.dtype)


class Block(nn.Module):

    def __init__(self,dim,num_heads,lpapas_ratio,qkv_bias):
        super(Block,self).__init__()

        self.attention_layer = attention_layer(dim,num_heads,lpapas_ratio,qkv_bias)

        self.forward_liner = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU6(),
        )

    def forward(self,x,adj,mask):

        x = F.layer_norm(x,[x.shape[-1]])

        out_1 = F.layer_norm(x + F.layer_norm(self.attention_layer(x,adj,mask),[x.shape[-1]]),[x.shape[-1]])

        out = F.layer_norm(out_1 + self.forward_liner(out_1), [x.shape[-1]])

        return out


class graph_transformer(nn.Module):

    def __init__(self,conf):
        super(graph_transformer,self).__init__()

        self.num_layer = conf.trans_layer

        self.block_list = nn.ModuleList()
        for i in range(self.num_layer):

            self.block_list.append(Block(conf.hidden_dim,conf.pool_transformer_heads,conf.lpapas_ratio, conf.qkv_bias))


    def forward(self,x,adj,mask):

        for i in range(self.num_layer):

            x = self.block_list[i](x,adj,mask)

        return x



class GA (nn.Module):
    def __init__(self,input_dim):
        super(GA, self).__init__()
        self.h_gate = nn.Sequential(nn.Linear(input_dim,64),nn.Linear(64,8),nn.Linear(8,1))

    def forward(self, x):

        h_gate_sco = F.softmax(self.h_gate(x),dim=1)
        return self.mean_readout(h_gate_sco * x)


    def mean_readout(self, H):
        return torch.mean(H, dim=1)

    def sum_readout(self, H):
        return torch.sum(H, dim=1)


# GCN basic operation
class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn=0, add_self=0, normalize_embedding=0,
                 dropout=0.0, relu=0, bias=True):
        super(GCNBlock, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        self.relu = relu
        self.bn = bn
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        if self.bn:
            self.bn_layer = torch.nn.BatchNorm1d(output_dim)

        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None

    def forward(self, x, adj, mask):
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        if self.bn:
            index = mask.sum(dim=1).long().tolist()
            bn_tensor_bf = mask.new_zeros((sum(index), y.shape[2]))
            bn_tensor_af = mask.new_zeros(*y.shape)
            start_index = []
            ssum = 0
            for i in range(x.shape[0]):
                start_index.append(ssum)
                ssum += index[i]
            start_index.append(ssum)
            for i in range(x.shape[0]):
                bn_tensor_bf[start_index[i]
                    :start_index[i+1]] = y[i, 0:index[i]]
            bn_tensor_bf = self.bn_layer(bn_tensor_bf)
            for i in range(x.shape[0]):
                bn_tensor_af[i, 0:index[i]
                             ] = bn_tensor_bf[start_index[i]:start_index[i+1]]
            y = bn_tensor_af
        if self.dropout > 0.001:
            y = self.dropout_layer(y)
        if self.relu == 'relu':
            y = torch.nn.functional.relu(y)
        elif self.relu == 'lrelu':
            y = torch.nn.functional.leaky_relu(y, 0.1)
        return y


class Dense_GIN_layer(nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GINConv`."""
    def __init__(self,nn,eps: float = 0.0, train_eps: bool = False,
    ):
        super(Dense_GIN_layer,self).__init__()

        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.eps.data.fill_(self.initial_eps)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     for name,parm in self.nn


    def forward(self, x, adj, mask = None,
                add_loop: bool = True):

        B, N, _ = adj.size()

        out = torch.matmul(adj, x)
        if add_loop:
            out = (1 + self.eps) * x + out

        out = self.nn(out)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

class Dense_GIN(nn.Module):

    def __init__(self,num_layer,input_dim,output_dim,hidden_dim=0,eps: float = 0.0,train_eps: bool = False,):
        super(Dense_GIN, self).__init__()

        self.num_layer = num_layer

        self.model = nn.ModuleList()

        if self.num_layer == 1:

            emdedimg_mlp = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=output_dim,bias=True),
                nn.ReLU6()
            )
            self.model.append(Dense_GIN_layer(emdedimg_mlp))
        elif self.num_layer == 2:
            emdedimg_mlp = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_dim,bias=True),
                nn.ReLU6()
            )
            out_mlp =  nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=output_dim,bias=True),
                nn.ReLU6()
            )
            self.model.append(Dense_GIN_layer(emdedimg_mlp))
            self.model.append(Dense_GIN_layer(out_mlp))
        else:
            emdedimg_mlp = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_dim,bias=True),
                nn.ReLU6()
            )
            hidden_mlp = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim,bias=True),
                nn.ReLU6()
            )
            out_mlp = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=output_dim,bias=True),
                nn.ReLU6()
            )
            self.model.append(Dense_GIN_layer(emdedimg_mlp))
            for i in range(1,self.num_layer-1):
                self.model.append(Dense_GIN_layer(hidden_mlp))
            self.model.append(Dense_GIN_layer(out_mlp))
        # self.reset_parameters()


    def forward(self,x,adj,mask):

        for i in range(self.num_layer):
            x = self.model[i](x,adj,mask)

        return x

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y



class Att_scores(nn.Module):
    def __init__(self,dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0):
        super(Att_scores,self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, X):
        B, N, C = X.shape
        qkv = self.qkv(X).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn)

        scores = torch.sum(torch.sum(attn, dim=1), dim=-1)

        return scores

class Att_cls_scores(nn.Module):
    def __init__(self,dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0):
        super(Att_cls_scores,self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, dim], requires_grad=True))

    def forward(self, X):


        expand_cls_embedding = self.cls_embedding.expand(X.shape[0], 1, -1)
        X = torch.cat([X, expand_cls_embedding], dim=1)
        B, N, C = X.shape
        qkv = self.qkv(X).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]

        attn = (q @ k.transpose(-2, -1)[:, :, :, 1:]) * self.scale
        attn = self.attn_drop(attn[:, :, 1, :])
        scores = torch.sum(attn, dim=1)

        return scores


"""transfoemer 模块"""
class transformer_insight(nn.Module):
    def __init__(self,d_model=128, nhead=4, dim_feedforward=512, transformer_dropout=0.3,
                 transformer_activation='relu', num_encoder_layers=4,  transformer_norm_input=False):
        super(transformer_insight, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, transformer_dropout, transformer_activation
        )
        encoder_norm = nn.LayerNorm(d_model, elementwise_affine=True).cuda()
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm).cuda()
        if transformer_norm_input:
            self.norm_input = nn.LayerNorm(d_model, elementwise_affine=True).cuda()

    def forward(self, data, mask):

        if self.norm_input is not None:
            data = self.norm_input(data)
        data = data.transpose(0,1)
        out = self.transformer(data, src_key_padding_mask=mask)

        return out.transpose(0,1)


class DegreePickBlock(nn.Module):
    def __init__(self, config):
        super(DegreePickBlock, self).__init__()
        self.filt_percent = config.percent
        self.assign_ratio = config.diffPool_assign_ratio    # diffpool module assign ratio
        self.max_node_num = config.diffPool_max_num_nodes   # maximum node number in dataset
        self.inter_channel_gcn = InterChannelGCN(config.hidden_dim, config.hidden_dim)

        self.aggregation = config.aggregation_1

        self.att_norm_X = config.att_norm_X  # 是否使用得分对X进行归一化
        self.trans_X_1 = config.trans_X_1
        self.channel_1 = config.Channel_1
        self.get_att_score = config.get_att_score

        if self.channel_1 == 'transformer':
            self.argation = transformer_insight(d_model=config.hidden_dim, nhead=config.transformer_head_number,
                                                dim_feedforward=config.transformer_forward,
                                                transformer_dropout=config.transformer_dropout,
                                                transformer_activation="relu",
                                                num_encoder_layers=config.transformer_layer_number,
                                                transformer_norm_input=config.transformer_norm_input)
            self.w = nn.Parameter(torch.zeros(config.hidden_dim))
        elif self.channel_1 == 'bilstm':
            self.argation = nn.LSTM(input_size=config.hidden_dim, hidden_size=config.hidden_dim,
                                    num_layers=config.bilstm_layer_number, bias=config.bilstm_bias,
                                    batch_first=True, bidirectional=True, dropout=config.bilstm_drop_out)
            self.w = nn.Parameter(torch.zeros(int(config.hidden_dim * 2)))
        elif self.channel_1 == 'lstm':
            self.argation = nn.LSTM(input_size=config.hidden_dim, hidden_size=config.hidden_dim,
                                    num_layers=config.bilstm_layer_number, bias=config.bilstm_bias,
                                    batch_first=True, bidirectional=False, dropout=config.bilstm_drop_out)
            self.w = nn.Parameter(torch.zeros(config.hidden_dim))
        elif self.channel_1 == 'bigru':
            self.argation = nn.GRU(input_size=config.hidden_dim, batch_first=True, hidden_size=config.hidden_dim,
                                   num_layers=config.gru_layer_number, bidirectional=True, dropout=config.gru_drop_out)
            self.w = nn.Parameter(torch.zeros(int(config.hidden_dim * 2)))
        elif self.channel_1 == 'gru':
            self.argation = nn.GRU(input_size=config.hidden_dim, batch_first=True, hidden_size=config.hidden_dim,
                                   num_layers=config.gru_layer_number, bidirectional=False, dropout=config.gru_drop_out)
            self.w = nn.Parameter(torch.zeros(config.hidden_dim))
        elif self.channel_1 == 'g-unet':
            self.w = nn.Parameter(torch.zeros(config.hidden_dim))
        elif self.channel_1 == 'gcn':
            self.argation = GCNBlock(config.hidden_dim, 1,
                                             config.bn, config.gcn_res, config.gcn_norm, config.dropout, config.relu).cuda()
        if self.channel_1 != 'gcn' and self.channel_1 != 'degrees':
            torch.nn.init.normal_(self.w)


    def forward(self, X, adj, mask, assign_matrix, H_coarse):
        '''
        input:
            X:  node input features , [batch,node_num,input_dim],dtype=float
            adj: adj matrix, [batch,node_num,node_num], dtype=float
            mask: mask for nodes, [batch,node_num]
            assign_matrix: assign matrix in diffpool module, [batch, node_num, cluster_num]
            H_coarse: embedding matrix of coarse graph, [batch, cluster_num, hidden_dim], dtype=float
        outputs:
            out: unormalized classification prob, [batch,hidden_dim]
            H: batch of node hidden features, [batch,node_num,pass_dim]
            new_adj: pooled new adj matrix, [batch, k_max, k_max]
            new_mask: [batch, k_max]
        '''

        k_max = int(math.ceil(self.filt_percent * adj.shape[-1]))
        k_list = [int(math.ceil(self.filt_percent * x)) for x in mask.sum(dim=1).tolist()]

        # for inter_channel convolution
        cluster_num = int(self.max_node_num * self.assign_ratio)

        """确定要使用哪种方式"""
        if self.channel_1 == 'degrees':
            scores = adj.sum(dim=-1)
        elif self.channel_1 == 'gru' or self.channel_1 == 'bigru' or self.channel_1 == 'lstm' or self.channel_1 == 'bilstm':
            hidden, _ = self.argation(X)
            if self.get_att_score == 'w':
                scores = torch.matmul(hidden, self.w)
            elif self.get_att_score == 'sum':
                scores = torch.sum(hidden, dim=-1)
            elif self.get_att_score == 'mean':
                scores = torch.mean(hidden, dim=-1)
            if self.trans_X_1:
                X = hidden
        elif self.channel_1 == 'gcn':
            scores = self.argation(X, adj, mask)

        elif self.channel_1 == 'transformer':
            hidden = self.argation(X, mask=mask)
            if self.get_att_score == 'w':
                scores = torch.matmul(hidden, self.w)
            elif self.get_att_score == 'sum':
                scores = torch.sum(hidden, dim=-1)
            elif self.get_att_score == 'mean':
                scores = torch.mean(hidden, dim=-1)
            if self.trans_X_1:
                X = hidden
        elif self.channel_1 == 'g-unet':
            scores = torch.matmul(X, self.w)

        _, top_index = torch.topk(scores, k_max, dim=1)
        top_index = top_index.reshape(top_index.shape[0], -1)
        """是否将得分映射到特征上"""
        if self.att_norm_X:
            scores = F.normalize(scores, dim=-1)
            scores = scores.reshape(scores.shape[0], -1, 1)
            X = X * scores

        # for update embedding and adjacency matrix
        new_mask = X.new_zeros(X.shape[0], k_max)
        S_reserve = X.new_zeros(X.shape[0], k_max, adj.shape[-1])
        H_tmp= X.new_zeros(X.shape[0], k_max, X.shape[-1])
        inter_channel_adj = X.new_zeros(X.shape[0], k_max, cluster_num)


        if H_coarse  is not None:
            for i, k in enumerate(k_list):
                new_mask[i][0:k] = 1
                inter_channel_adj[i, 0:k] = assign_matrix[i, top_index[i, :k]]
                S_reserve[i, 0:k] = adj[i, top_index[i, :k]]
                S_reserve[i,:,top_index[i, :k]] = 0
                H_tmp[i, 0:k] = X[i, top_index[i, :k]]

            if self.aggregation > 0:
                H = self.aggregation * torch.matmul(S_reserve, X) + H_tmp
            else:
                H = H_tmp
            H = self.inter_channel_gcn(H, H_coarse, inter_channel_adj)
        else:
            for i, k in enumerate(k_list):
                new_mask[i][0:k] = 1
                S_reserve[i, 0:k] = adj[i, top_index[i, :k]]
                S_reserve[i, :, top_index[i, :k]] = 0
                H_tmp[i, 0:k] = X[i, top_index[i, :k]]

            if self.aggregation > 0:
                H = self.aggregation * torch.matmul(S_reserve, X) + H_tmp
            else:
                H = H_tmp

        new_adj = torch.matmul(torch.matmul(S_reserve, adj), torch.transpose(S_reserve, 1, 2))

        return top_index, H, k_list, new_adj, new_mask

class AttPoolBlock(nn.Module):
    def __init__(self, config):
        super(AttPoolBlock, self).__init__()
        self.gcn = GCNBlock(config.hidden_dim, config.hidden_dim, config.bn, config.gcn_res, config.gcn_norm, config.dropout, config.relu)
        self.inter_channel_gcn = InterChannelGCN(config.hidden_dim, config.hidden_dim)
        self.filt_percent = config.percent
        self.assign_ratio = config.diffPool_assign_ratio # diffpool module assign ratio
        self.max_node_num = config.diffPool_max_num_nodes   # maximum node number in dataset

        self.get_att_score = config.get_att_score  # 计算得分的方式是使用一个权重矩阵进行特征变换，还是直接求和
        self.att_norm_X = config.att_norm_X  # 是否使用得分对X进行归一化
        self.trans_X_3 = config.trans_X_3
        self.channel_3 = config.Channel_3
        self.aggregation = config.aggregation_3
        if self.channel_3 == 'transformer':
            self.argation = transformer_insight(d_model=config.hidden_dim, nhead=config.transformer_head_number,
                                                dim_feedforward=config.transformer_forward,
                                                transformer_dropout=config.transformer_dropout,
                                                transformer_activation="relu",
                                                num_encoder_layers=config.transformer_layer_number,
                                                transformer_norm_input=config.transformer_norm_input)
            self.w = nn.Parameter(torch.zeros(config.hidden_dim))

        elif self.channel_3 =='attention':
            self.argation = Att_scores(dim=config.hidden_dim,num_heads=config.transformer_head_number,attn_drop=config.transformer_dropout)
            self.w = nn.Parameter(torch.zeros(config.hidden_dim))
        elif self.channel_3 =='attention_cls':
            self.argation = Att_cls_scores(dim=config.hidden_dim,num_heads=config.transformer_head_number,attn_drop=config.transformer_dropout)
            self.w = nn.Parameter(torch.zeros(config.hidden_dim))
        elif self.channel_3 == 'bilstm':
            self.argation = nn.LSTM(input_size=config.hidden_dim, hidden_size=config.hidden_dim,
                                    num_layers=config.bilstm_layer_number, bias=config.bilstm_bias,
                                    batch_first=True, bidirectional=True, dropout=config.bilstm_drop_out)
            self.w = nn.Parameter(torch.zeros(int(config.hidden_dim * 2)))
        elif self.channel_3 == 'lstm':
            self.argation = nn.LSTM(input_size=config.hidden_dim, hidden_size=config.hidden_dim,
                                    num_layers=config.bilstm_layer_number, bias=config.bilstm_bias,
                                    batch_first=True, bidirectional=False, dropout=config.bilstm_drop_out)
            self.w = nn.Parameter(torch.zeros(config.hidden_dim))
        elif self.channel_3 == 'bigru':
            self.argation = nn.GRU(input_size=config.hidden_dim, batch_first=True, hidden_size=config.hidden_dim,
                                   num_layers=config.gru_layer_number, bidirectional=True, dropout=config.gru_drop_out)
            self.w = nn.Parameter(torch.zeros(int(config.hidden_dim * 2)))
        elif self.channel_3 == 'gru':
            self.argation = nn.GRU(input_size=config.hidden_dim, batch_first=True, hidden_size=config.hidden_dim,
                                   num_layers=config.gru_layer_number, bidirectional=False, dropout=config.gru_drop_out)
            self.w = nn.Parameter(torch.zeros(config.hidden_dim))
        elif self.channel_3 == 'g-unet':
            self.w = nn.Parameter(torch.zeros(config.hidden_dim))
        elif self.channel_3 == 'readout':
            self.w = nn.Parameter(torch.zeros(config.hidden_dim, config.hidden_dim))
        self.w_degree = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        torch.nn.init.normal_(self.w)
        torch.nn.init.kaiming_uniform_(self.w_degree)

    def forward(self, X, adj, mask, assign_matrix, H_coarse):
        '''
        input:
            X:  node input features , [batch,node_num,input_dim],dtype=float
            adj: adj matrix, [batch,node_num,node_num], dtype=float
            mask: mask for nodes, [batch,node_num]
            assign_matrix: assign matrix in diffpool module, [batch, node_num, next_layer_node_num]
            H_coarse: embedding matrix of coarse graph, [batch, cluster_num, hidden_dim], dtype=float
        outputs:
            out: unormalized classification prob, [batch,hidden_dim]
            H: batch of node hidden features, [batch,node_num,pass_dim]
            new_adj: pooled new adj matrix, [batch, k_max, k_max]
            new_mask: [batch, k_max]
        '''

        k_max = int(math.ceil(self.filt_percent * adj.shape[-1]))
        k_list = [int(math.ceil(self.filt_percent * x)) for x in mask.sum(dim=1).tolist()]
        degree = torch.sum(adj,dim=-1).reshape(adj.shape[0],-1,1)
        arr_degree = degree * self.w_degree
        X = X + arr_degree


        # 确定要使用哪种方式
        if self.channel_3 == 'gru' or self.channel_3 == 'bigru' or self.channel_3 == 'lstm' or self.channel_3 == 'bilstm':
            hidden, _ = self.argation(X)
            if self.get_att_score == 'w':
                scores = torch.matmul(hidden, self.w)
            elif self.get_att_score == 'sum':
                scores = torch.sum(hidden, dim=-1)
            elif self.get_att_score == 'mean':
                scores = torch.mean(hidden, dim=-1)
            if self.trans_X_3:
                X = hidden

        elif self.channel_3 == 'transformer':
            hidden = self.argation(X, mask=mask)
            if self.get_att_score == 'w':
                scores = torch.matmul(hidden, self.w)
            elif self.get_att_score == 'sum':
                scores = torch.sum(hidden, dim=-1)
            else:
                scores = torch.mean(hidden, dim=-1)
            if self.trans_X_3:
                X = hidden
        elif self.channel_3 == 'attention':
            scores = self.argation(X)

        elif self.channel_3 == 'g-unet':
            scores = torch.matmul(X, self.w)
        elif self.channel_3 == 'readout':
            hidden = self.readout(X)
            reference_hidden = F.relu(torch.matmul(hidden, self.w))
            reference_hidden = reference_hidden.unsqueeze(1)
            inner_prod = torch.mul(X, reference_hidden).sum(dim=-1)
            scores = F.softmax(inner_prod, dim=1)

        _, top_index = torch.topk(scores, k_max, dim=1)
        if self.att_norm_X:
            scores = F.normalize(scores, dim=-1)
            scores = scores.reshape(scores.shape[0], -1, 1)
            X = X * scores

        # for update embedding and adjacency matrix
        new_mask = X.new_zeros(X.shape[0], k_max)
        S_reserve = X.new_zeros(X.shape[0], k_max, adj.shape[-1])

        # for inter-channel convolution
        cluster_num = int(self.max_node_num * self.assign_ratio)
        H_tmp = X.new_zeros(X.shape[0], k_max, X.shape[-1])
        inter_channel_adj = X.new_zeros(X.shape[0], k_max, cluster_num)

        if H_coarse is not None:
            for i, k in enumerate(k_list):
                new_mask[i][0:k] = 1
                inter_channel_adj[i, 0:k] = assign_matrix[i, top_index[i, :k]]
                S_reserve[i, 0:k] = adj[i, top_index[i, :k]]
                S_reserve[i, :, top_index[i, :k]] = 0
                H_tmp[i, 0:k] = X[i, top_index[i, :k]]

            if self.aggregation > 0:
                H = self.aggregation * torch.matmul(S_reserve, X) + H_tmp
            else:
                H = H_tmp
            H = self.inter_channel_gcn(H, H_coarse, inter_channel_adj)

        else:
            for i, k in enumerate(k_list):
                new_mask[i][0:k] = 1
                S_reserve[i, 0:k] = adj[i, top_index[i, :k]]
                S_reserve[i, :, top_index[i, :k]] = 0
                H_tmp[i, 0:k] = X[i, top_index[i, :k]]

            if self.aggregation > 0:
                H = self.aggregation * torch.matmul(S_reserve, X) + H_tmp
            else:
                H = H_tmp

        new_adj = torch.matmul(torch.matmul(S_reserve, adj), torch.transpose(S_reserve, 1, 2))
        return top_index, H, k_list, new_adj,new_mask

    def readout(self, x):
        return x.sum(dim=1)

#Inter-channel GCN Block
class InterChannelGCN(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=True, normalize=False):
        super(InterChannelGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.add_self = add_self
        self.normalize = normalize
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim)).cuda()
        nn.init.xavier_normal_(self.weight)

    def forward(self, H_fine, H_coarse, inter_channel_adj):
        out = torch.matmul(inter_channel_adj, H_coarse)
        if self.add_self:
            out += H_fine
        out = torch.matmul(out, self.weight)
        out = F.relu(out)
        if self.normalize:
            out = F.normalize(out)
        return out

class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1

        self.bias = True
        if args is not None:
            self.bias = args.diffPool_bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
        pred_input_dim = hidden_dim * (2 + len(self.conv_block))
        self.transform = self.build_pred_layers(pred_input_dim, [], hidden_dim)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias)
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''
        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        #out_all = []
        #out, _ = torch.max(x, dim=1)
        #out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x,adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        embedding_mask = embedding_mask.unsqueeze(-1)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out,_ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x,adj)
        #x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        #print(output.size())
        return ypred

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1,1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args)
        add_self = not concat
        self.num_pooling = num_pooling

        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        assign_dim = int(max_num_nodes * assign_ratio)
        self.assign_conv_first, self.assign_conv_block, self.assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)
        assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
        self.assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)
        self.gcn_after_pooling = GraphConv(input_dim, hidden_dim, add_self=add_self, normalize_embedding=True)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x, adj, mask):
        embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last, mask)
        assign_matrix = self.gcn_forward(x, adj,
                self.assign_conv_first, self.assign_conv_block, self.assign_conv_last, mask)

        # [batch_size x num_nodes x next_lvl_num_nodes]
        embedding_tensor = self.transform(embedding_tensor)
        assign_matrix = nn.Softmax(dim=-1)(self.assign_pred(assign_matrix))
        mask = mask.unsqueeze(-1)
        if mask is not None:
            assign_matrix = assign_matrix * mask

        # update pooled features and adj matrix
        x_new = torch.matmul(torch.transpose(assign_matrix, 1, 2), embedding_tensor)
        adj_new = torch.transpose(assign_matrix, 1, 2) @ adj @ assign_matrix
        mask_new = torch.ones(size=[mask.shape[0],assign_matrix.shape[-1]],device=mask.device,requires_grad=mask.requires_grad,dtype=mask.dtype)

        return assign_matrix, x_new, embedding_tensor, adj_new,mask_new


class FIPool(nn.Module):
    def __init__(self, args):
        super(FIPool, self).__init__()
        self.args = args
        self.degreePick = DegreePickBlock(args)
        self.AttPool = AttPoolBlock(args)
        self.DiffPool = SoftPoolingGcnEncoder(args.diffPool_max_num_nodes, args.hidden_dim, args.hidden_dim, args.hidden_dim,
            args.diffPool_num_classes, args.diffPool_num_gcn_layer, args.hidden_dim, assign_ratio=args.diffPool_assign_ratio,
            num_pooling=args.diffPool_num_pool, bn=args.diffPool_bn, dropout=args.diffPool_dropout)
        if args.readout == 'mean':
            self.readout = self.mean_readout
        elif args.readout == 'sum':
            self.readout = self.sum_readout

    def forward(self, X, adj, mask):
        '''
        input:
            X:  node input features , [batch,node_num,input_dim],dtype=float
            adj: adj matrix, [batch,node_num,node_num], dtype=float
            mask: mask for nodes, [batch,node_num]
        outputs:
            out:unormalized classification prob, [batch,hidden_dim]
            H: batch of node hidden features, [batch,node_num,pass_dim]
            new_adj: pooled new adj matrix, [batch, k_max, k_max]
            new_mask: [batch, k_max]
        '''
        # result of DiffPool model
        assign_matrix, H_coarse,_,_,_ = self.DiffPool(X, adj, mask)

        if self.args.local_topology and self.args.with_feature and self.args.global_topology:
            degree_based_index, H1, k_list1, _,_ = self.degreePick.forward(X, adj, mask, assign_matrix, H_coarse)
            feature_based_index, H2, k_list2,_,_  = self.AttPool(X, adj, mask, assign_matrix, H_coarse)

            index1 = [x[:k] for x, k in zip(degree_based_index.tolist(), k_list1)]
            index2 = [y[:k] for y, k in zip(feature_based_index.tolist(), k_list2)]
            intersection_index = [list(set(x) & set(y)) for x, y in zip(index1, index2)]
            union_index = [list(set(x) | set(y)) for x, y in zip(index1, index2)]
        elif not self.args.local_topology and self.args.with_feature and self.args.global_topology:
            H1 = None
            feature_based_index, H2, k_list2,_,_  = self.AttPool(X, adj, mask, assign_matrix, H_coarse)

            index1 = []
            index2 = [y[:k] for y, k in zip(feature_based_index.tolist(), k_list2)]
            intersection_index = index1
            union_index = index2
        elif self.args.local_topology and not self.args.with_feature and self.args.global_topology:
            degree_based_index, H1, k_list1, _,_ = self.degreePick.forward(X, adj, mask, assign_matrix, H_coarse)
            H2 = None

            index1 = [x[:k] for x, k in zip(degree_based_index.tolist(), k_list1)]
            index2 = []
            intersection_index = index2
            union_index = index1
        elif self.args.local_topology and self.args.with_feature and not self.args.global_topology:
            H_coarse = H_coarse / H_coarse
            degree_based_index, H1, k_list1, _,_ = self.degreePick.forward(X, adj, mask, assign_matrix, H_coarse)
            feature_based_index, H2, k_list2, _,_ = self.AttPool(X, adj, mask, assign_matrix, H_coarse)

            index1 = [x[:k] for x, k in zip(degree_based_index.tolist(), k_list1)]
            index2 = [y[:k] for y, k in zip(feature_based_index.tolist(), k_list2)]
            intersection_index = [list(set(x) & set(y)) for x, y in zip(index1, index2)]
            union_index = [list(set(x) | set(y)) for x, y in zip(index1, index2)]

        k_list = [len(x) for x in union_index]
        k_max = max(k_list)

        # for update embedding and adjacency matrix
        new_mask = X.new_zeros(X.shape[0], k_max)
        S_reserve = X.new_zeros(X.shape[0], k_max, adj.shape[-1])

        for i, k in enumerate(k_list):
            new_mask[i][0:k] = 1
            S_reserve[i, 0:k] = adj[i, union_index[i]]

        # update feature matrix and adjacency matrix
        new_adj = torch.matmul(torch.matmul(S_reserve, adj), torch.transpose(S_reserve, 1, 2))
        new_H = self.reconstruct_feature_matrix(H1, H2, index1, index2, union_index, intersection_index, k_max, k_list)
        return new_H, new_adj, new_mask

    def reconstruct_feature_matrix(self, H1, H2, index1, index2, union_index, intersection_index, k_max, k_list):
        difference_set1 = [list(set(x) - set(y)) for x, y in zip(index1, intersection_index)]
        difference_set2 = [list(set(x) - set(y)) for x, y in zip(index2, intersection_index)]
        if H1 is not None and H2 is not None:
            new_H = H1.new_zeros(H1.shape[0], k_max, H1.shape[-1])
            for i, k in enumerate(k_list):
                # idx1 = [index1[i].index(x) for x in difference_set1[i]]
                # idx_common = [union_index[i].index(x) for x in intersection_index[i]]
                # idx2 = [index2[i].index(x) for x in difference_set2[i]]

                idx_common_new = [union_index[i].index(x) for x in intersection_index[i]]
                idx_common_origin1 = [index1[i].index(x) for x in intersection_index[i]]
                idx_common_origin2 = [index2[i].index(x) for x in intersection_index[i]]

                idx_new_1 = [union_index[i].index(x) for x in difference_set1[i]]
                idx1 = [index1[i].index(x) for x in difference_set1[i]]

                idx_new_2 = [union_index[i].index(x) for x in difference_set2[i]]
                idx2 = [index2[i].index(x) for x in difference_set2[i]]

                new_H[i, idx_common_new] = (H1[i, idx_common_origin1] + H2[i, idx_common_origin2]) / 2
                new_H[i, idx_new_1] = H1[i, idx1]
                new_H[i, idx_new_2] = H2[i, idx2]
        elif H1 is not None and H2 is None:
            new_H = H1
        elif H1 is None and H2 is not None:
            new_H = H2

        return new_H


    def mean_readout(self, H):
        return torch.mean(H, dim=1)

    def sum_readout(self, H):
        return torch.sum(H, dim=1)


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers=2, dropout=0., indrop=0):
        super(MLPClassifier, self).__init__()

        self.num_layers = num_layers
        if self.num_layers == 2:
            self.h1_weights = nn.Linear(input_size, hidden_size)
            self.h2_weights = nn.Linear(hidden_size, num_class)
            torch.nn.init.xavier_normal_(self.h1_weights.weight.t())
            torch.nn.init.constant_(self.h1_weights.bias, 0)
            torch.nn.init.xavier_normal_(self.h2_weights.weight.t())
            torch.nn.init.constant_(self.h2_weights.bias, 0)
        elif self.num_layers == 1:
            self.h1_weights = nn.Linear(input_size, num_class)
            torch.nn.init.xavier_normal_(self.h1_weights.weight.t())
            torch.nn.init.constant_(self.h1_weights.bias, 0)
        self.dropout = dropout
        self.indrop = indrop
        if self.dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, y=None):
        if self.indrop and self.dropout > 0.001:
            x = self.dropout_layer(x)
        if self.num_layers == 2:
            h1 = self.h1_weights(x)
            if self.dropout > 0.001:
                h1 = self.dropout_layer(h1)
            h1 = F.relu(h1)

            logits = self.h2_weights(h1)
        elif self.num_layers == 1:
            logits = self.h1_weights(x)

        softmax_logits = F.softmax(logits, dim=1)
        logits = F.log_softmax(logits, dim=1)


        return logits
