import torch
import time
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error,accuracy_score
import numpy as np
from utils import result_file,convert_seconds
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os.path as osp
import os
from graph_model import LT,FIPool_Net
class Trainer_tudataset(object):
    def __init__(self,conf):

        self.conf = conf

        self.find_num_exams()
        if conf.model == "LT":
            self.model = LT(self.conf).cuda()
        elif conf.model == "FIPool":
            self.model = FIPool_Net(self.conf,conf.embeding_method).cuda()
        if conf.optimizer == "Adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)

    def find_num_exams(self):
        path = osp.join(osp.dirname(osp.abspath(__file__)),"..","tensorbord_logs",self.conf.dataset_name)
        if osp.exists(path):
            logs_listdir = os.listdir(path)
            self.num_exams = len(logs_listdir) + 1
        else:
            self.num_exams = 1
        self.logs_path = osp.join(path, str(self.num_exams))



    def run(self,Graphs):

        final_dict = {}
        final_dict_list = {}

        self.tb_writer = SummaryWriter(log_dir=self.logs_path)
        start = time.time()
        for i in range(self.conf.fold):

            for name,parm in self.model.named_parameters():
                if 'weight' in name:
                    torch.nn.init.normal_(parm,mean=0,std=0.01)

            torch.cuda.reset_accumulated_memory_stats()
            print("fold:" + str(i+1))

            self.fold_idx = i
            Graphs.use_fold_data(self.fold_idx)
            self.train_dataloader = DataLoader(dataset=Graphs.train_graphs, shuffle=True, batch_size=self.conf.batch_size, drop_last=False)
            self.test_dataloader = DataLoader(dataset=Graphs.test_graphs, shuffle=False, batch_size=self.conf.batch_size, drop_last=False)


            best_val_dict = self.train()

            if i == 0:
                for j in best_val_dict.keys():
                    final_dict_list[j] = []
            for j in best_val_dict.keys():
                final_dict_list[j].append(best_val_dict[j])
            self.tb_writer.add_scalars("fold_result",
                                  {j: best_val_dict[j] for j in best_val_dict.keys()},self.fold_idx+1)

        self.tb_writer.close()
        for j in final_dict_list.keys():
            final_dict[j] = sum(final_dict_list[j])/self.conf.fold

        self.tb_writer.add_scalars("Result",
                                   {j: final_dict[j] for j in final_dict.keys()})


        final_print = 'Result: '
        for i in final_dict.keys():
            final_print = final_print + i + ":" + str(final_dict[i])[:5] + "\t\t\t"

        use_time = time.time()-start
        final_print = final_print + "time_cost:"+ convert_seconds(use_time)
        print(final_print)
        result_file(self.conf, final_dict, self.num_exams,use_time)

    def train(self):

        best_epoch = 0
        plot_li_train = []
        plot_li_eval = []
        best_val_print = ''
        acc = 0
        best_val_dict = {}

        for epoch in range(self.conf.epoch):
            loss_li = []
            start = time.time()
            y_true = []
            y_pred = []
            self.model.train()
            for batch in tqdm(self.train_dataloader, desc='Training progress', unit='graphs'):

                x,adj,mask,labels = batch[0].cuda(),batch[1].cuda(),batch[2].cuda(),batch[3].reshape(-1).cuda()

                out = self.model(x,adj,mask)
                out = out[0]


                if "MAE" in self.conf.eval_indicators:
                    loss = F.mse_loss(out.reshape([-1]), labels.reshape([-1]))
                else:
                    loss = F.cross_entropy(out, labels)


                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                pred = out.data.max(1)[1].cpu().numpy()
                y_pred.append(pred)
                y_true.append(labels.cpu().numpy())
                loss = loss.item()
                loss_li.append(loss)
            train_loss = sum(loss_li)/len(loss_li)


            train_acc_dict = self.eval_method(y_true=y_true,y_pred=y_pred)
            train_dict = train_acc_dict
            train_dict["loss"] = train_loss
            plot_li_train.append(train_acc_dict)
            train_eval_print = 'Result: '
            for i in train_acc_dict.keys():
                train_eval_print = train_eval_print + i + ":" + str(train_acc_dict[i])[:5] + "\t\t\t"
            end = time.time()

            print(' \033[1;36mTrain epoch {} :    Loss: {:.5f}     time cost: {:.2f}s\033[0m'.format(epoch+1, train_loss, end - start))
            print('\033[1;36m' + train_eval_print+ "\033[0m")
            train_acc_dict.pop("loss")
            val_feature,val_acc_dict, val_loss, val_time = self.test(self.test_dataloader)
            val_dict = val_acc_dict
            val_dict["loss"] = val_loss
            plot_li_eval.append(val_dict)

            val_eval_print = 'Result: '
            for i in val_acc_dict.keys():
                val_eval_print = val_eval_print + i + ":" + str(val_acc_dict[i])[:5] + "\t\t\t"

            if val_acc_dict["ACC"] > acc:
                acc = val_acc_dict["ACC"]
                best_epoch = epoch + 1
                best_val_dict = val_acc_dict
                best_val_print = "Best_val Result: " + "Best_epoch:" + str(best_epoch) + "\t\t\t"+ val_eval_print[7:]
                best_val_feature = val_feature

            print('\033[1;34mVal epoch {} result:  Loss: {:.5f}     time cost: {:.2f}s\033[0m'.format(epoch + 1, val_loss, end - start))
            print('\033[1;34m' + val_eval_print + "\033[0m")
            print('\033[1;32m' + best_val_print + "\033[0m")


            self.tb_writer.add_scalars('Loss' ,
                                  {'%s_train_Loss' % (str(self.fold_idx + 1)): train_loss,
                                                '%s_test_Loss' % (str(self.fold_idx + 1)): val_loss,},epoch+1)

            self.tb_writer.add_scalars('Epoch_result',
                                  {'%s_train_%s' % (str(self.fold_idx + 1),j): train_acc_dict[j] for j in train_acc_dict.keys()},epoch+1)
            val_acc_dict.pop("loss")
            self.tb_writer.add_scalars('Epoch_result',
                                       {'%s_test_%s' % (str(self.fold_idx + 1), j): val_acc_dict[j] for j in
                                        val_acc_dict.keys()}, epoch)
        self.tb_writer.add_embedding(tag="fold:" + str(self.fold_idx) + "_test_tsen", mat=best_val_feature,
                                     metadata=self.label)

        return best_val_dict


    def test(self,data):
        self.model.eval()
        loss_li = []
        start = time.time()
        y_true = []
        y_pred = []
        feature_li = []
        with torch.no_grad():
            for batch in tqdm(data, desc='Valid/Test progress', unit='graphs'):

                x,adj,mask,labels = batch[0].cuda(),batch[1].cuda(),batch[2].cuda(),batch[3].reshape(-1).cuda()

                out = self.model(x,adj,mask)
                feature_li.append(out[1])
                out = out[0]


                if "MAE" in self.conf.eval_indicators:
                    loss = F.mse_loss(out.reshape([-1]), labels.reshape([-1]))
                else:
                    loss = F.cross_entropy(out, labels)
                pred = out.data.max(1)[1].cpu().numpy()
                y_pred.append(pred)
                y_true.append(labels.cpu().numpy())
                loss = loss.item()
                loss_li.append(loss)
            loss = sum(loss_li) / len(loss_li)
            feature = torch.stack(feature_li[:-1]).view(-1,self.conf.hidden_dim)
            feature = torch.cat([feature,feature_li[-1]],dim=0)
            eval_dict = self.eval_method(y_true=y_true, y_pred=y_pred)
            end = time.time()

        return feature, eval_dict, loss, end-start

    def list2np(self,li):

        out = np.stack(li[:-1]).reshape(-1)
        out = np.concatenate([out, li[-1]], axis=-1)
        return out



    def eval_method(self,y_true,y_pred):

        y_pred = self.list2np(y_pred)
        y_true = self.list2np(y_true)
        self.label = y_true

        out = {}

        if "MAE" in self.conf.eval_indicators:
            out["MAE"] = mean_absolute_error(y_true=y_true,y_pred=y_pred)
        if "F1" in self.conf.eval_indicators:
            out["F1"] = f1_score(y_true, y_pred)*100
        if "ROC" in self.conf.eval_indicators:
            out["ROC"] = roc_auc_score(y_true, y_pred)*100
        if "ACC" in self.conf.eval_indicators:
            out["ACC"] = accuracy_score(y_true=y_true, y_pred=y_pred)*100

        return out






