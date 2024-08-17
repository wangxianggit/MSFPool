import torch
import torch.nn as nn
from graph_modules import GCN_Dense,Dense_GIN_layer,Dense_GIN,Dense_GCN_layer,graph_transformer,GCNBlock,FIPool,MLPClassifier,GA

class LT(nn.Module):

    def __init__(self,config):
        super(LT,self).__init__()

        self.local_pool_layer = config.local_pool_layer
        self.trans_layer = config.trans_layer
        self.aggration_method = config.aggration_method
        self.dropout = nn.Dropout(config.dropout)
        self.local_topy = config.local_topy
        self.global_topy = config.global_topy

        if config.embeding_method == "GIN":
            self.embeding = Dense_GIN(config.embeding_layer,config.num_features,config.hidden_dim,config.hidden_dim)
        elif config.embeding_method == "GCN":
            self.embeding = GCN_Dense(input_dim=config.num_features,output_dim=config.hidden_dim,hidden_dim=config.hidden_dim,num_layers=config.embeding_layer)
        self.pool_key = config.pool_key
        if self.pool_key:

            self.conv_model= nn.ModuleList()
            for i in range(config.local_pool_layer):
                if self.local_topy:
                    if config.pool_conv_method == "GCN":
                        self.conv_model.append(Dense_GCN_layer(in_channels=config.hidden_dim,out_channels=config.hidden_dim))
                    elif config.pool_conv_method == "GIN":
                        gin_mlp = nn.Linear(in_features=config.hidden_dim,out_features=config.hidden_dim,bias=True)
                        self.conv_model.append(Dense_GIN_layer(nn=gin_mlp))

            if self.aggration_method == "BILSTM":

                self.aggration = nn.LSTM((config.hidden_dim * config.local_pool_layer),
                                             config.hidden_dim, bidirectional=True, batch_first=True)

            elif self.aggration_method == "LSTM":
                self.aggration = nn.LSTM((config.hidden_dim * config.local_pool_layer),
                                         (config.hidden_dim), bidirectional=False, batch_first=True)

            elif self.aggration_method == "GRU":
                self.aggration = nn.GRU((config.hidden_dim * config.local_pool_layer),
                                         (config.hidden_dim), bidirectional=False, batch_first=True)

            elif self.aggration_method == "BIGRU":
                self.aggration = nn.GRU((config.hidden_dim * config.local_pool_layer) ,
                                         (config.hidden_dim ), bidirectional=True, batch_first=True)

            self.alpha = nn.Parameter(torch.Tensor([0.5]))
            if self.global_topy and config.trans_layer > 0 :
                self.trans_pool = graph_transformer(config)

        self.mlp = nn.Linear(config.hidden_dim, config.num_task)
        self.action_method = nn.ReLU()



    def forward(self, x, adj, mask):

        embed = self.embeding(x, adj, mask)
        feature_conv_li = []

        if self.pool_key:
            if self.local_topy and self.local_pool_layer > 0:
                out_1 = embed
                for i in range(self.local_pool_layer):
                    out_1 = self.action_method(self.conv_model[i](out_1, adj, mask))
                    out_1 = self.dropout(out_1)
                    feature_conv_li.append(out_1)

                if self.aggration_method == "BILSTM" or self.aggration_method == "LSTM" or self.aggration_method == "BIGRU" or self.aggration_method ==  "GRU":

                    feature_conv = torch.stack(feature_conv_li,dim=-1).view(x.shape[0],x.shape[1],-1)
                    att, hc = self.aggration(feature_conv)
                    if self.aggration_method == "BILSTM" or self.aggration_method == "LSTM":
                        h,c = hc[0].transpose(0,1),hc[1].transpose(0,1)
                        feature_conv = torch.cat([h,c],dim=1).sum(dim=1)
                    else:
                        feature_conv = hc.transpose(0,1).sum(dim=1)
                else:
                    feature_conv = out_1.sum(dim=1)

            if self.global_topy and self.trans_layer > 0 :
                feature_trans = self.trans_pool(embed,adj,mask).sum(dim=1)

            if self.global_topy and self.trans_layer > 0 and  self.local_topy and self.local_pool_layer > 0:
                out = feature_trans * self.alpha + feature_conv * (1-self.alpha)
            elif self.global_topy and self.trans_layer > 0:
                out = feature_trans
            elif self.local_topy and self.local_pool_layer > 0:
                out = feature_conv
            else:
                out = embed
        else:
            out = embed.sum(dim=1)

        out_1 = self.mlp(out)

        return out_1,out




class FIPool_Net(nn.Module):
    def __init__(self, args, convolution_method):
        super(FIPool_Net, self).__init__()
        self.args = args
        self.hierarchical_num = args.hierarchical_num
        embed_method = convolution_method
        self.embeds = nn.ModuleList()
        if embed_method == 'GCN':
            self.embed = GCNBlock(args.num_features, args.hidden_dim, args.bn, args.gcn_res, args.gcn_norm, args.dropout, args.relu)
            for i in range(self.hierarchical_num):
                self.embeds.append(GCNBlock(args.hidden_dim, args.hidden_dim, args.bn, args.gcn_res, args.gcn_norm, args.dropout, args.relu))
        elif embed_method == 'GIN':
            self.embed = Dense_GIN(num_layer=args.gin_layer,input_dim=args.num_features,output_dim=args.hidden_dim,hidden_dim=args.hidden_dim)
            for i in range(self.hierarchical_num):
                self.embeds.append(
                    Dense_GIN(num_layer=args.gin_layer, input_dim=args.hidden_dim, output_dim=args.hidden_dim,
                              hidden_dim=args.hidden_dim))
        self.muchPools = nn.ModuleList()
        for i in range(self.hierarchical_num):
            self.muchPools.append(FIPool(args))
        if args.readout == 'mean':
            self.readout = self.mean_readout
        elif args.readout == 'sum':
            self.readout = self.sum_readout
        elif args.readout == 'GA':
            self.readout = GA(input_dim=args.hidden_dim)
        elif  args.readout == 'FC':
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(in_features=args.hidden_dim, out_features=args.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=args.hidden_dim, out_features=args.hidden_dim),
                torch.nn.ReLU(),
            )
            # self.mlpc = MLPClassifier(input_size=args.hidden_dim * (args.hierarchical_num + 1), hidden_size=args.hidden_dim, num_class=args.num_class)
        self.mlpc = MLPClassifier(input_size=args.hidden_dim, hidden_size=args.hidden_dim,
                                  num_class=args.num_class)
        # self.norm_X = nn.LayerNorm(args.hidden_dim)

    def forward(self, xs, adjs, masks):
        H = self.embed(xs, adjs, masks)
        Z = torch.zeros_like(self.readout(H))
        # Z = self.readout(H)
        for i in range(self.hierarchical_num):
            H = self.embeds[i](H, adjs, masks)
            H, adjs, masks = self.muchPools[i](H, adjs, masks)
            Z = Z + self.readout(H)

            # Z = torch.cat([Z+self.readout(H)],dim=-1)

        logits = self.mlpc(Z)
        return logits,Z

    def mean_readout(self, H):
        return torch.mean(H, dim=1)

    def sum_readout(self, H):
        return torch.sum(H, dim=1)
