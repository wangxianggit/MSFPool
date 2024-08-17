from torch_geometric.datasets  import TUDataset
from torch_geometric.utils import degree
from torch_geometric.transforms import OneHotDegree
import torch
from torch.utils.data import Dataset,DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
import os.path as osp
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np


class Graph_dataset(Dataset):

    def __init__(self,x,adj,mask,y):
        self.x = x
        self.adj = adj
        self.mask = mask
        self.y = y

    def __getitem__(self, item):
        return self.x[item],self.adj[item],self.mask[item],self.y[item]

    def __len__(self):
        return self.x.size(0)


class TU_utill(object):

    def __init__(self,config):

        self.conf = config
        self.dataset_loader()
        config.num_task = self.num_task
        config.num_features = self.num_features



    def get_num_nodes(self,dataset):

        num_nodes = []
        for g in dataset:
            num_nodes.append(g.num_nodes)

        return num_nodes


    def sprese2dense(self,data,max_num_nodes):

        if data.edge_attr is None:
            edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)
        else:
            edge_attr = data.edge_attr

        size = torch.Size([max_num_nodes, max_num_nodes] + list(edge_attr.size())[1:])
        adj = torch.sparse_coo_tensor(data.edge_index, edge_attr, size)

        data.adj = adj.to_dense()
        data.edge_index = None
        data.edge_attr = None

        data.mask = torch.zeros(max_num_nodes, dtype=torch.bool)
        data.mask[:data.num_nodes] = 1

        if data.x is not None:
            size = [max_num_nodes - data.x.size(0)] + list(data.x.size())[1:]
            data.x = torch.cat([data.x, data.x.new_zeros(size)], dim=0)

        return data



    def random_split_10_fold(self):
        self.dir_name = osp.join(osp.dirname(osp.abspath(__file__)),"10fold_idx",self.conf.dataset_name,str(self.conf.fold)+"fold")

        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
            skf = StratifiedKFold(n_splits=self.conf.fold, shuffle=True, random_state=self.conf.seed)
            labels = self.graphs.y
            self.idx_list = list(skf.split(np.zeros(len(self.graphs)), labels))
            for i in range(len(self.idx_list)):
                np.savetxt(osp.join(self.dir_name,str(i)+"_train.csv"),self.idx_list[i][0],fmt="%d",delimiter=",")
                np.savetxt(osp.join(self.dir_name, str(i) + "_test.csv"), self.idx_list[i][1], fmt="%d",
                           delimiter=",")

    def use_fold_data(self, fold_idx):
        self.fold_idx = fold_idx

        train_idx = np.loadtxt(osp.join(self.dir_name,str(self.fold_idx)+"_train.csv"),int,delimiter=",")
        test_idx = np.loadtxt(osp.join(self.dir_name,str(self.fold_idx)+"_test.csv"),int,delimiter=",")
        self.train_graphs = [self.graphs[i] for i in train_idx]
        self.test_graphs = [self.graphs[i] for i in test_idx]




    def dataset_loader(self):

        dataset_name = self.conf.dataset_name


        if "ogb" in dataset_name or "Tox21" in dataset_name:
            if "Tox21" in dataset_name :
                train_dataset = TUDataset(root="./datasets", name=dataset_name + "_training")
                val_dataset = TUDataset(root="./datasets", name=dataset_name + "_evaluation")
                test_dataset = TUDataset(root="./datasets", name=dataset_name + "_testing")
                self.num_task = train_dataset.num_classes
                self.num_features = train_dataset.num_node_features
            elif "ogb" in dataset_name:
                dataset = PygGraphPropPredDataset(name=dataset_name, root='dataset/')
                split_idx = dataset.get_idx_split()
                train_dataset = dataset[split_idx["train"]]
                val_dataset = dataset[split_idx["valid"]]
                test_dataset = dataset[split_idx["test"]]
                self.num_task = dataset.num_classes
                self.num_features = dataset.num_node_features
            else:
                train_dataset = []
                val_dataset = []
                test_dataset = []

            if train_dataset:
                dataset_full = []
                for i in [train_dataset, val_dataset, test_dataset]:
                    num_node_list = self.get_num_nodes(i)
                    max_num_nodes = max(num_node_list)
                    x_list = []
                    adj_list = []
                    mask_list = []
                    y_list = []
                    for g in i:
                        data = self.sprese2dense(g, max_num_nodes)
                        x_list.append(data.x.float())
                        adj_list.append(data.adj)
                        mask_list.append(data.mask)
                        y_list.append(data.y)

                    x = torch.stack(x_list)
                    adj = torch.stack(adj_list)
                    if adj.dim() == 4:
                        adj = torch.sum(adj, dim=-1)
                    mask = torch.stack(mask_list)
                    y = torch.stack(y_list)

                    data = Graph_dataset(x, adj, mask, y)
                    dataset_full.append(data)

                    self.train_data = DataLoader(dataset=dataset_full[0], shuffle=True, batch_size=self.conf.batch_size, drop_last=False)
                    self.val_data = DataLoader(dataset=dataset_full[1], shuffle=False, batch_size=self.conf.batch_size, drop_last=False)
                    self.test_data = DataLoader(dataset=dataset_full[2], shuffle=False, batch_size=self.conf.batch_size, drop_last=False)


        else:
            if "ZINC" in dataset_name:
                dataset = TUDataset(root="./datasets", name=dataset_name+"_full")
            else:
                dataset = TUDataset(root="./datasets", name=dataset_name)
            if dataset[0].x is None:
                degrees = degree(dataset.edge_index[0], dtype=torch.long)
                max_num_edges = torch.max(degrees)
                dataset = TUDataset(root="./datasets", name=dataset_name, transform=OneHotDegree(max_num_edges))
                print('Use node degree for node features')

            self.num_task = dataset.num_classes
            self.num_features = dataset.num_node_features

            num_node_list = self.get_num_nodes(dataset)
            max_num_nodes = max(num_node_list)
            x_list = []
            adj_list = []
            mask_list = []
            y_list = []
            for g in dataset:
                data = self.sprese2dense(g, max_num_nodes)
                x_list.append(data.x.float())
                adj_list.append(data.adj)
                mask_list.append(data.mask)
                y_list.append(data.y)

            x = torch.stack(x_list)
            if dataset_name == "DD":
                adj = adj_list
            else:
                adj = torch.stack(adj_list)
                if adj.dim() == 4:
                    adj = torch.sum(adj, dim=-1)
            mask = torch.stack(mask_list)
            y = torch.stack(y_list)

            self.graphs = Graph_dataset(x, adj, mask, y)
            self.random_split_10_fold()


            print('# ================== Dataset %s Information ==================' % dataset_name)
            print('# total classes: %d' % self.num_task)
            print('# node feature: %d' % self.num_features)







if __name__ == '__main__':

    from config import Graph_Config

    conf = Graph_Config(osp.join(osp.dirname(osp.abspath(__file__)), 'config', 'NCI1.ini'))
    a = TU_utill(conf)
    a.use_fold_data(1)
    train_data = DataLoader(dataset=a.train_graphs, shuffle=True, batch_size=32, drop_last=False)
    for i in train_data:
        print(i)
        break





