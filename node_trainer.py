import torch
from node_model import GCN, LT
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, accuracy_score
import os.path as osp
import os
import time
from torch.utils.tensorboard import SummaryWriter
from utils import result_file

class trainer(object):

    def __init__(self, conf):

        self.conf = conf
        self.find_num_exams()
        if self.conf.model == "LT":
            self.model = LT(conf).cuda()
        else:
            self.model = GCN(conf).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)


    def find_num_exams(self):
        self.logs_path = osp.join(osp.dirname(osp.abspath(__file__)), "..", "tensorbord_logs", self.conf.dataset_name, self.conf.model)
        model_path = osp.join(osp.dirname(osp.abspath(__file__)), 'model_save', self.conf.task, self.conf.dataset_name, self.conf.model)
        if osp.exists(model_path):
            model_listdir = os.listdir(model_path)
            self.num_exams = len(model_listdir) + 1
        else:
            self.num_exams = 1
            os.makedirs(model_path)
        if not osp.exists(self.logs_path):
            os.makedirs(self.logs_path)

        self.model_path = osp.join(model_path, str(self.num_exams) + ".pt")


    def run(self, data):
        start = time.time()
        self.tb_writer = SummaryWriter(log_dir=self.logs_path)
        self.data = data.cuda()
        self.train()
        test_acc = self.eval()
        self.tb_writer.close()
        end = time.time()
        result_file(self.conf,test_acc,self.num_exams,end-start)

    def train(self):
        best_val_loss = 1e10
        best_epoch = 1
        train_mask = self.data.train_mask
        val_mask = self.data.val_mask
        for i in range(self.conf.epoch):
            start = time.time()
            out = self.model(self.data)
            loss = F.cross_entropy(out[train_mask], self.data.y[train_mask])
            loss.backward()
            train_loss = loss.item()

            self.optimizer.step()
            self.optimizer.zero_grad()
            pred = out.data.max(1)[1].cpu().numpy()
            train_acc, train_print = self.eval_method(pred[train_mask.cpu()], self.data.y[train_mask.cpu()].cpu())
            end = time.time()

            print(' \033[1;36mepoch {} : \033[0m'.format(i + 1))
            print(' \033[1;36mTrain ' + train_print + '    Loss {:.5f}    time cost {:.2f}s\033[0m'.format(train_loss, end - start))
            start = time.time()
            with torch.no_grad():
                val_loss = F.cross_entropy(out[val_mask], self.data.y[val_mask])
                val_acc, val_print = self.eval_method(pred[val_mask.cpu()], self.data.y[val_mask.cpu()].cpu())

                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    best_epoch = i + 1
                    torch.save(self.model, self.model_path)

            end = time.time()
            print(' \033[1;33mVal ' + val_print + '    Loss {:.5f}    time cost {:.2f}s\033[0m'.format(val_loss, end - start))
            print(' \033[1;32mbest epoch: {}      best val loss: {:.5f}\033[0m'.format(best_epoch, best_val_loss))


            self.tb_writer.add_scalars('{}_Loss'.format(self.num_exams),
                                  {'train_Loss': train_loss,
                                                'val_Loss': val_loss,}, i+1)
            self.tb_writer.add_scalars('{}_Epoch_Result'.format(self.num_exams),{'Train_%s' % j: train_acc[j] for j in train_acc.keys()}, i + 1)
            self.tb_writer.add_scalars('{}_Epoch_Result'.format(self.num_exams), {'Val_%s' % j: val_acc[j] for j in val_acc.keys()}, i + 1)





    def eval(self):
        start = time.time()
        model = torch.load(self.model_path)
        model.eval()
        test_mask = self.data.test_mask
        with torch.no_grad():
            out = model(self.data)
            test_loss = F.cross_entropy(out[test_mask], self.data.y[test_mask])
            pred = out.data.max(1)[1].cpu().numpy()
            test_acc, test_print = self.eval_method(pred[test_mask.cpu()], self.data.y[test_mask.cpu()].cpu())
            end = time.time()
            print(' \033[1;31mTest ' + test_print + '    Loss {:.5f}    time cost {:.2f}s\033[0m'.format(test_loss, end - start))
            self.tb_writer.add_scalars('Final Result', {'Test_%s' % j: test_acc[j] for j in test_acc.keys()}, self.num_exams)
            del self.conf.eval_indicators
            self.tb_writer.add_hparams(self.conf.__dict__, test_acc)
        return test_acc


    def eval_method(self, y_true, y_pred):

        out = {}

        if "MAE" in self.conf.eval_indicators:
            out["MAE"] = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        if "F1" in self.conf.eval_indicators:
            out["F1"] = f1_score(y_true, y_pred) * 100
        if "ROC" in self.conf.eval_indicators:
            out["ROC"] = roc_auc_score(y_true, y_pred) * 100
        if "ACC" in self.conf.eval_indicators:
            out["ACC"] = accuracy_score(y_true=y_true, y_pred=y_pred) * 100

        eval_print = 'Result:\t\t\t'
        for i in out.keys():
            eval_print = eval_print + i + ":" + str(out[i])[:5] + "\t\t\t"

        return out, eval_print
