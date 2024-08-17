import os
import argparse
import torch
import numpy as np
import random
import os.path as osp
from config import Graph_Config

def set_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)




if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--task', default='graph_classification', type=str, help='graph_classification')
    parse.add_argument('--dataset', default='PROTEINS', type=str, help='Cora,PubMed,CiteSeer')
    parse.add_argument('--seed', default=42, type=str, help='seed')
    args = parse.parse_args()
    set_seed(args.seed)

    config_file = osp.join(osp.dirname(osp.abspath(__file__)), 'config', '%s.ini' % args.dataset)
    from node_dataset_utils import dataset_process
    from node_trainer import trainer
    from graph_dataset_util import TU_utill
    from graph_trainer import Trainer_tudataset

    conf = Graph_Config(config_file)
    data = TU_utill(config=conf)
    trainer = Trainer_tudataset(conf)

    trainer.run(data)


