import time
import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils.plot import *
from utils.spot import *
from utils.pot import *
from utils.utils import *
from utils.diagnosis import *
from model.Fransformer import DTE_AD
from data_factory.data_loader import get_loader_segment
from torch.backends import cudnn


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        #data load
        self.train_loader, self.ori_train, self.ori_test = get_loader_segment(self.data_path,
                                                                              batch_size=self.batch_size,
                                                                              win_size=self.win_size, mode='train',
                                                                              dataset=self.dataset)
        self.vali_loader, self.ori_train, self.ori_test = get_loader_segment(self.data_path, batch_size=self.batch_size,
                                                                             win_size=self.win_size, mode='val',
                                                                             dataset=self.dataset)
        self.test_loader, self.ori_train, self.ori_test = get_loader_segment(self.data_path, batch_size=self.batch_size,
                                                                             win_size=self.win_size, mode='test',
                                                                             dataset=self.dataset)
        self.thre_loader, self.ori_train, self.ori_test = get_loader_segment(self.data_path, batch_size=self.batch_size,
                                                                             win_size=self.win_size,
                                                                             mode='thre', dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss(reduction = 'mean')
        self.path = self.model_save_path + experiment

        # pot_eval에서 쓰이는 값 (percentile, ret)
        lm_d = {
        'SMD': [(0.984, 1.04), (0.99995, 1.06)],
        'SWaT': [(1.1, 1), (0.993, 1)],
        'SMAP': [(2.6, 1), (0.976, 1)],#0.977
        'MSL': [(1.28, 1), (0.999, 1.04)],
        'PSM': [(3.5, 1.04), (0.961, 1)],
        'MSDS': [(6.1, 1), (0.9, 1.04)]}
  
        if 'machine' in self.dataset:
            self.lm = lm_d['SMD'][0]
        else:
            self.lm = lm_d[self.dataset][0]


    def build_model(self):
        self.model = DTE_AD(feats=self.input_c, n_window=self.win_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_v = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, time_attn, t_attn, f_attn = self.model(input)

            loss = self.criterion(output, input)
            loss_v.append((loss).item())


            #val_loss = [loss_item * attn for loss_item, attn in zip(loss_v, time_attn)]
            #val_loss_np = [item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item for item in val_loss]


            #np.average(loss_v) -> original
            #torch.mean(val_loss)
        return np.average(loss_v), self.optimizer.param_groups[0]['lr']

    def train(self):

        print("======================TRAIN MODE======================")

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        start = time.time()
        accuracy_list = []
        for epoch in range(self.num_epochs):
            iter_count = 0
            loss_list = []
            epoch_time = time.time()
            self.model.train()

            for i, (input_data, labels) in enumerate(self.train_loader):
                iter_count += 1
                input = input_data.float().to(self.device)

                output, time_attn, t_attn, f_attn = self.model(input)

                loss = self.criterion(output, input)
                
                
                loss_list.append((loss).item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - start) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    start = time.time()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss_list) # -> original
            loss_tensor = torch.tensor(loss_list)

            #print(f"loss_tensor = {loss_tensor.shape}\ntime_attn = P{time_attn.shape}")
            #attn_loss = [loss_item * attn for loss_item, attn in zip(loss_list, time_attn)]
            #attn_loss_np = [item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item for item in attn_loss]
            #train_loss = torch.mean(attn_loss)
            #train_loss = np.average(attn_loss_np)
            #train_loss = loss_tensor * time_attn


            vali_loss, lr = self.vali(self.vali_loader)
            #ls = vali_loss.detach().cpu().numpy()
            accuracy_list.append((vali_loss, lr))

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, self.path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time.time() - start) + ' s' + color.ENDC)
        plot_accuracies(accuracy_list, f'{self.dataset}')

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)
        # loss 계산할 때 각 데이터 포인트에 대한 손실을 개별적으로 반환하도록함 (anomaly score == reconstruction error)

        # (1) stastic on the train set
        #train_energy = []
        train_attn = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            #print(f'iterate ----------------------------> {i}')
            input = input_data.float().to(self.device)
            output, time_attn, t_attn, f_attn = self.model(input)

            loss = criterion(output, input)
            loss = loss.detach().cpu().numpy()
            train_attn.append(loss)

            time_loss = 0.0

            #loss = loss.detach().cpu().numpy()
            #train_energy.append(loss)

            #print(f'loss_item len = {len(train_energy)}\ntime_attn len = {len(time_attn)}')
            #print(f'loss_item = {train_energy[0].shape}\ntime_attn = {time_attn[0].shape}')


            '''for n in range(len(time_attn)):
                #time_e = time_attn[n].detach().cpu().numpy()
                time_loss += time_attn[n]
                

            time_loss = time_loss / len(time_attn)'''
            

            #t_loss = torch.mean(torch.sum(time_loss, dim=-1), dim=1)
            #time_loss_a = t_loss.unsqueeze(-1)
            #time_loss_a = time_loss_a.detach().cpu().numpy()
            '''
            t_loss = torch.mean(torch.sum(t_attn, dim=-1), dim=1)
            time_loss_a = t_loss.unsqueeze(-1)
            time_loss_a = time_loss_a.detach().cpu().numpy()
            
            f_loss = torch.mean(torch.sum(t_attn, dim=-1), dim=1)
            freq_loss_a = f_loss.unsqueeze(-1)
            freq_loss_a = freq_loss_a.detach().cpu().numpy()
            

            #print(f'loss_item = {loss.shape}\ntime_loss_a = {time_loss_a.shape}')

            train_loss = loss * (0.5 * time_loss_a + 0.5 * freq_loss_a)

            train_attn.append(train_loss)'''

        #train_loss_t = [loss_item * attn.detach().cpu().numpy() for loss_item, attn in zip(train_energy, time_attn)]

        #train_loss_a = np.average(train_loss_t)

        train_energy = np.concatenate(train_attn, axis=0)
        train_energy = train_energy.reshape(-1, self.input_c)

        #train_energy = np.concatenate(train_energy, axis=0)
        #train_energy = train_energy.reshape(-1, self.input_c)

        #print(f'train_energy = {train_energy.shape}')

        # (2) evaluation on the test set
        test_labels = []
        #test_energy = []
        test_attn = []
        recon_test = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            test_labels.append(labels)
            output, time_attn, t_attn, f_attn = self.model(input)

            loss = criterion(output, input)

            loss = loss.detach().cpu().numpy()

            output = output.detach().cpu().numpy()
            recon_test.append(output)

            time_loss = 0.0
            feat = loss.shape[2]

            '''for n in range(len(time_attn)):
                time_loss += time_attn[n]

            time_loss = time_loss / len(time_attn)'''

            t_attn_ = torch.where(t_attn > 0, t_attn, torch.tensor(1e-9).to(self.device))

            t_loss = torch.log(t_attn_)

            f_attn_ = torch.where(f_attn > 0, f_attn, torch.tensor(1e-9).to(self.device))
            
            f_loss = torch.log(f_attn_)

            
            attn = t_loss 
            attn = torch.mean(torch.sum(attn, dim=-1), dim=1) 
            attn = attn.unsqueeze(-1)
            attn = attn.detach().cpu().numpy()

            #test_loss = loss * -attn
            test_loss = -attn
            test_loss = np.repeat(test_loss, feat, axis=2)
            print(f'loss len = {loss.shape}')
            print(f'attn len = {attn.shape}')
            test_loss = np.where(test_loss > 0, test_loss, 0) 


            test_attn.append(test_loss)


        # label
        test_labels = np.concatenate(test_labels, axis=0)
        test_labels = test_labels.reshape(-1, self.input_c)

        # anomaly score
        attens_energy = np.concatenate(test_attn, axis=0)
        attens_energy = attens_energy.reshape(-1, self.input_c)


        # test-set reconstruction results
        recon_test = np.concatenate(recon_test, axis=0)
        recon_test = recon_test.reshape(-1, self.input_c)

        #attens_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        attens_energy = np.array(attens_energy)

        print(f'lm = {self.lm}')
        print(f'dataset = {self.dataset}')

        df = pd.DataFrame()
        preds = []
        thresh = []

        # 데이터의 각 변수마다 결과 도출
        for i in range(self.input_c):
            lt, l, ls = train_energy[:, i], attens_energy[:, i], test_labels[:, i]
            result, pred, thres = pot_eval(lt, l, ls, self.lm)
            preds.append(pred)
            thresh.append(thres)
            df = df.append(result, ignore_index=True)

        # reconstruction 결과(test data, reconstructed data), detection 결과(anomaly score, threshold)
        plotter_o(f'{self.dataset}', self.ori_test, recon_test, attens_energy, test_labels, thresh)

        # 변수별 결과의 평균 도출
        lossTfinal, lossFinal = np.mean(train_energy, axis=1), np.mean(attens_energy, axis=1)
        labelsFinal = (np.sum(test_labels, axis=1) >= 1) + 0
        result, _, _ = pot_eval(lossTfinal, lossFinal, labelsFinal, self.lm)
        result.update(hit_att(attens_energy, test_labels))
        result.update(ndcg(attens_energy, test_labels))
        
        print(f"lossFinal{lossFinal}")

        print(df)
        pprint(result)

        