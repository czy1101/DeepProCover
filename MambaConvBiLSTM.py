# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import lightgbm as lgb
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

from copy import deepcopy
from time import time
from joblib import Parallel, delayed
from scipy.stats.mstats import gmean
from bayes_opt import BayesianOptimization
from triqler.qvality import getQvaluesFromScores

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
import warnings
import math
from mamba_ssm import Mamba2
from mamba_ssm import Mamba
warnings.filterwarnings('ignore')
#plt.rcParams['font.sans-serif'] = 'Arial'


def showcols(df):
    cols = df.columns.tolist()
    cols = cols + (10 - len(cols) % 10) % 10 * ['None']
    cols = np.array(cols).reshape(-1, 10)
    return pd.DataFrame(data=cols, columns=range(1, 11))


def Prob2PEP(y, y_prob):
    label = np.array(deepcopy(y))
    score = np.array(deepcopy(y_prob))
    srt_idx = np.argsort(-score)
    label_srt = label[srt_idx]
    score_srt = score[srt_idx]
    targets = score_srt[label_srt == 1]
    decoys = score_srt[label_srt != 1]
    _, pep = getQvaluesFromScores(targets, decoys, includeDecoys=True)
    return pep[np.argsort(srt_idx)]


def Score2Qval(y0, y_score):
    y = np.array(deepcopy(y0)).flatten()
    y_score = np.array(y_score).flatten()
    y[y != 1] = 0
    srt_idx = np.argsort(-y_score)
    y_srt = y[srt_idx]
    cum_targets = y_srt.cumsum()
    cum_decoys = np.abs(y_srt - 1).cumsum()
    FDR = np.divide(cum_decoys, cum_targets)
    qvalue = np.zeros(len(FDR))
    qvalue[-1] = FDR[-1]
    qvalue[0] = FDR[0]
    for i in range(len(FDR) - 2, 0, -1):
        qvalue[i] = min(FDR[i], qvalue[i + 1])
    qvalue[qvalue > 1] = 1
    return qvalue[np.argsort(srt_idx)]


def GroupProteinPEP2Qval(data, file_column, protein_column, target_column, pep_column):
    data['Protein_label'] = data[protein_column] + '_' + data[target_column].map(str)
    df = []
    for i, j in data.groupby(file_column):
        df_pro = j[['Protein_label', target_column, pep_column]].sort_values(pep_column).drop_duplicates(
            subset='Protein_label', keep='first')
        df_pro['protein_qvalue'] = Score2Qval(df_pro[target_column].values, -df_pro[pep_column].values)
        df.append(pd.merge(j,
                           df_pro[['Protein_label', 'protein_qvalue']],
                           on='Protein_label',
                           how='left'))
    return pd.concat(df, axis=0).drop('Protein_label', axis=1)

 

class IonCoding:
    def __init__(self, bs=1000, n_jobs=-1):
        ino = [
            '{0}{2};{1}{2};{0}{2}(2+);{1}{2}(2+);{0}{2}-NH3;{1}{2}-NH3;{0}{2}(2+)-NH3;{1}{2}(2+)-NH3;{0}{2}-H20;{1}{2}-H20;{0}{2}(2+)-H20;{1}{2}(2+)-H20'.
            format('b', 'y', i) for i in range(1, 47)]
        ino = np.array([i.split(';') for i in ino]).flatten()
        self.MI0 = pd.DataFrame({'MT': ino})
        self.bs = bs
        self.n_jobs = n_jobs

    def fit_transfrom(self, data):
        print('++++++++++++++++OneHotEncoder CMS(Chage + Modified sequence)++++++++++++++')
        t0 = time()
        x = self.onehotfeature(data['CMS']).reshape(data.shape[0], -1, 30)

        print('using time', time() - t0)
        print('x shape: ', x.shape)
        print('++++++++++++++++++++++++++Construct Ion Intensities Array++++++++++++++++++++')
        t0 = time()
        y = self.ParallelinoIY(data[['Matches', 'Intensities']])
        print('using time', time() - t0)
        print('y shape: ', y.shape)
        return x, y

    def onehotfeature(self, df0, s=48):
        df = df0.apply(lambda x: x + (s - len(x)) * 'Z')
        # B: '_(ac)M(ox)'; 'J': '_(ac)'; 'O': 'M(ox)'; 'Z': None
        aminos = '123456ABCDEFGHIJKLMNOPQRSTVWYZ'
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(np.repeat(np.array(list(aminos)), s).reshape(-1, s))
        seqs = np.array(list(df.apply(list)))


        return enc.transform(seqs).toarray().reshape(df.shape[0], -1, 30)

    def ParallelinoIY(self, data):
        datasep = [data.iloc[i * self.bs: (i + 1) * self.bs] for i in range(data.shape[0] // self.bs + 1)]

        paraY = Parallel(n_jobs=self.n_jobs)(delayed(self.dataapp)(i) for i in datasep)


        return np.vstack(paraY).reshape(data.shape[0], -1, 12)

    def inoIY(self, x):
        MI = pd.DataFrame({'MT0': x[0].split(';'), 'IY': [float(i) for i in x[1].split(';')]})
        dk = pd.merge(self.MI0, MI, left_on='MT', right_on='MT0', how='left').drop('MT0', axis=1)
        dk.loc[dk.IY.isna(), 'IY'] = 0
        dk['IY'] = dk['IY'] / dk['IY'].max()
        return dk['IY'].values

    def dataapp(self, data):
        return np.array(list(data.apply(self.inoIY, axis=1)))


class CNN_BiLSTM(nn.Module):
    def __init__(self, batchsize, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(CNN_BiLSTM, self).__init__()
        self.cov = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,
                      out_channels=64,
                      kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5))
        #self.lstm = CustomBiLSTM(batchsize=batchsize, input_dim=64, hidden_dim=hidden_dim, num_layers=layer_dim,device=torch.device('cuda:0'))
        self.lstm = nn.LSTM(input_size=64,
                            hidden_size=hidden_dim,
                            num_layers=layer_dim,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.5)
        self.mambaBlock11 = Mamba(d_model=64, d_conv=3,d_state =16)

        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, output_dim),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.cov(x.permute(0, 2, 1))
        l_out=x.permute(0, 2, 1)
        #l_out = self.mambaBlock11(l_out)  # 46*64

        l_out, (l_hn, l_cn) = self.lstm(l_out, None)
 

    
        x = self.fc(l_out)
 

        return x


class MambaConvBiLSTM:
    def __init__(self, model=None, seed=0, test_size=0.2, lr=1e-3, l2=0.0,
                 batch_size=256, epochs=1000, nepoch=50, patience=50,
                 device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")):
        self.test_size = test_size
        self.seed = seed
        self.batch_size = batch_size
        self.device = device
        self.patience = patience
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.nepoch = nepoch
        self.model = model

    def fit(self, bkmsms):
        #bkmsms 标准msms
        print('+++++++++++++++++++++++++++Loading Trainset+++++++++++++++++++++')
        bkmsms['CMS'] = bkmsms['Charge'].map(str) + bkmsms['Modified sequence']
        bkmsms['CMS'] = bkmsms['CMS'].apply(lambda x: x.replace('_(ac)M(ox)', 'B').replace(
            '_(ac)', 'J').replace('M(ox)', 'O').replace('_', ''))

        bkmsms1 = self.selectBestmsms(bkmsms, s=100)[['CMS', 'Matches', 'Intensities']]
        x, y = IonCoding().fit_transfrom(bkmsms1)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.test_size, random_state=self.seed)
        y_true = np.array(y_val).reshape(y_val.shape[0], -1).tolist()
        train_db = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_db,
                                  batch_size=self.batch_size,
                                  num_workers=0,
                                  shuffle=True)


        val_db = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_db,
                                batch_size=self.batch_size,
                                num_workers=0,
                                shuffle=False)

        if self.model is None:
            torch.manual_seed(self.seed)
            #self.model = CNN_BiLSTM(30, 256, 2, 12)
            self.model = CNN_BiLSTM(self.batch_size, 30, 256, 2, 12, self.device)

        model = self.model.to(self.device)

        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2)

        val_losses = []
        val_cosines = []
        self.val_cosine_best = 0.0
        counter = 0

        print('+++++++++++++++++++DeepSpec Training+++++++++++++++++++++')
        for epoch in range(1, self.epochs + 1):
            for i, (x_batch, y_batch) in enumerate(train_loader):
                print(
                    f'current epoch: {epoch}, progress: {i}/{len(train_loader)} ({(i / len(train_loader)) * 100:.2f}%)')
                model.train()
                batch_x = x_batch.to(self.device)
                batch_y = y_batch.to(self.device)
                out = model(batch_x)

                loss = loss_func(out, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = 0
                y_valpreds = []
                for a, b in val_loader:
                    val_x = a.to(self.device)
                    val_y = b.to(self.device)
                    y_valpred = model(val_x)
                    y_valpreds.append(y_valpred)
                    val_loss += loss_func(y_valpred, val_y).item() / len(val_loader)

                val_losses.append(val_loss)
                y_valpreds = torch.cat([y_vp for y_vp in y_valpreds], dim=0)
                y_pred = np.array(y_valpreds.cpu()).reshape(y_val.shape[0], -1).tolist()

                val_cosine = self.cosine_similarity(y_true, y_pred)
                val_cosines.append(val_cosine)

                if val_cosine.mean() >= self.val_cosine_best:
                    counter = 0
                    self.val_cosine_best = val_cosine.mean()
                    self.val_loss_best = val_loss
                    self.bestepoch = epoch
                    torch.save(model, 'DeepSpec.pkl')
                else:
                    counter += 1

                if epoch % self.nepoch == 0 or epoch == self.epochs:
                    print(
                        '[{}|{}] val_loss: {} | val_cosine: {}'.format(epoch, self.epochs, val_loss, val_cosine.mean()))

                if counter >= self.patience:
                    print('EarlyStopping counter: {}'.format(counter))
                    break

        print('best epoch [{}|{}] val_ loss: {} | val_cosine: {}'.format(self.bestepoch, self.epochs,
                                                                         self.val_loss_best, self.val_cosine_best))
        self.traininfor = {'val_losses': val_losses, 'val_cosines': val_cosines}

    def predict(self, evidence, msms):
        dfdb = deepcopy(evidence)
        msms = deepcopy(msms).rename(columns={'id': 'Best MS/MS'})
        dfdb['CMS'] = dfdb['Charge'].map(str) + dfdb['Modified sequence']
        dfdb['CMS'] = dfdb['CMS'].apply(lambda x: x.replace('_(ac)M(ox)', 'B').replace(
            '_(ac)', 'J').replace('M(ox)', 'O').replace('_', ''))
        dfdb = pd.merge(dfdb, msms[['Best MS/MS', 'Matches', 'Intensities']], on='Best MS/MS', how='left')
        dfdb1 = deepcopy(dfdb[(~dfdb['Matches'].isna()) &
                              (~dfdb['Intensities'].isna()) &
                              (dfdb['Length'] <= 47) &
                              (dfdb['Charge'] <= 6)])[['id', 'CMS', 'Matches', 'Intensities']]
        print('after filter none Intensities data shape:', dfdb1.shape)#(1036043,4)
        print('+++++++++++++++++++Loading Testset+++++++++++++++++++++')
        x_test, y_test = IonCoding().fit_transfrom(dfdb1[['CMS', 'Matches', 'Intensities']])
        self.db_test = {'Data': dfdb1, 'x_test': x_test, 'y_test': y_test}
        x_test = torch.tensor(x_test, dtype=torch.float)
        test_loader = DataLoader(x_test,
                                 batch_size=self.batch_size,
                                 num_workers=0,
                                 shuffle=False)
        print('+++++++++++++++++++DeepSpec Testing+++++++++++++++++++++')
        y_testpreds = []
        model = torch.load('DeepSpec.pkl').to(self.device)

        model.eval()
        with torch.no_grad():
            for test_x in test_loader:
                test_x = test_x.to(self.device)
                y_testpreds.append(model(test_x))
            y_testpred = torch.cat(y_testpreds, dim=0)

        y_test = np.array(y_test).reshape(y_test.shape[0], -1)
        y_testpred = np.array(y_testpred.cpu())
        self.db_test['y_testpred'] = y_testpred
        y_testpred = y_testpred.reshape(y_test.shape[0], -1)
        CS = self.cosine_similarity(y_test, y_testpred)
        self.db_test['Cosine'] = CS
        output = pd.DataFrame({'id': dfdb1['id'].values, 'Cosine': CS})
        dfdb2 = pd.merge(dfdb, output, on='id', how='left')
        dfdb2['PEPCosine'] = dfdb2['Cosine'] / (1 + dfdb2['PEP'])
        dfdb2['ScoreCosine'] = dfdb2['Score'] / (1 + dfdb2['Cosine'])
        return dfdb2

    def cosine_similarity(self, y, y_pred):
        a, b = np.array(y), np.array(y_pred)
        res = np.array([[sum(a[i] * b[i]), np.sqrt(sum(a[i] * a[i]) * sum(b[i] * b[i]))]
                        for i in range(a.shape[0])])
        return np.divide(res[:, 0], res[:, 1])  # Cosine or DP
        # return 1 - 2 * np.arccos(np.divide(res[:, 0], res[:, 1])) / np.pi  # SA

    def selectBestmsms(self, df, lg=47, cg=6, s=100):
        return df[(df['Reverse'] != '+') & (~df['Matches'].isna()) &
                  (~df['Intensities'].isna()) & (df['Length'] <= lg) &
                  (df['Charge'] <= cg) & (df['Type'].isin(['MSMS', 'MULTI-MSMS']))
                  & (df['Score'] > s)].sort_values(
            'Score', ascending=False).drop_duplicates(
            subset='CMS', keep='first')[['CMS', 'Matches', 'Intensities']]

    def ValPlot(self):
        val_losses = self.traininfor['val_losses']
        val_cosines = self.traininfor['val_cosines']
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        lns1 = ax.plot(range(1, len(val_losses) + 1), val_losses,
                       color='orange', label='Val Loss={}'.format(round(self.val_loss_best, 5)))
        lns2 = ax.axvline(x=self.bestepoch, ls="--", c="b", label='kk')
        plt.xticks(size=15)
        plt.yticks(size=15)

        ax2 = ax.twinx()
        lns3 = ax2.plot(range(1, len(val_losses) + 1), [i.mean() for i in val_cosines],
                        color='red', label='Val Cosine={}'.format(round(self.val_cosine_best, 4)))

        lns = lns1 + lns3
        labs = [l.get_label() for l in lns]

        plt.yticks(size=15)
        ax.set_xlabel("Epoch", fontsize=18)
        ax.set_ylabel("Val Loss", fontsize=18)
        ax2.set_ylabel("Val Cosine", fontsize=18)
        ax.legend(lns, labs, loc=10, fontsize=15)
        plt.tight_layout()

    def CountNum(self, batchx):
        results = []
        for i in batchx:  # 256
            count = 0
            for ii in i:  

                if not (ii[-1] == 1 and torch.all(ii[:-1] == 0)):
                    count = count + 1
                else:
                    break
            results.append(count)

        return results

 
class CustomBiLSTM(nn.Module):
    def __init__(self, batchsize, input_dim, hidden_dim, num_layers, device, dropout=0.5):
        super(CustomBiLSTM, self).__init__()
        self.batchsize = batchsize
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = 1#num_layers
        self.lstm_layer = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                                  dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.mambaBlock1 = Mamba(d_model=self.batchsize, d_conv=2)

        self.reset_parameters()

    def lstm_forward(self, input, initial_states, w_ih, w_hh, b_ih, b_hh):
        """
        :param input:
        :param initial_states:初始状态
        :param w_ih:输入到隐藏层权重
        :param w_hh:隐藏层到隐藏层权重
        :param b_ih:输入到隐藏层偏置
        :param b_hh:隐藏层到隐藏层偏置
        :return:输出张量和最终的隐藏状态
        """

        h_0, c_0 = initial_states  # 初始状态
        batch_size, T, input_size = input.shape  # T 表示每一样本序列长度
        hidden_size = w_ih.shape[0] // 4
        prev_h = h_0
        prev_c = c_0
        batch_w_ih = w_ih.unsqueeze(0).tile(batch_size, 1, 1)  # [batch_size, 4*hidden_size, input_size]([2, 80, 10])
        batch_w_hh = w_hh.unsqueeze(0).tile(batch_size, 1, 1)  # [batch_size, 4*hidden_size, hidden_size]([2, 80, 20])

        output_size = hidden_size
        output = torch.zeros(batch_size, T, output_size)  # 输出序列[2,5,20]

        for t in range(T):
            x = input[:, t, :]  # 当前时刻的输入向量，[batch_size*input_size]
            w_times_x = torch.bmm(batch_w_ih, x.unsqueeze(-1))  # [batch_size, 4*hidden_size, 1],[2,80,1]
            w_times_x = w_times_x.squeeze(-1)  # [batch_size, 4*hidden_size],[2,80]

            w_times_h_prev = torch.bmm(batch_w_hh, prev_h.unsqueeze(-1))  # [batch_size, 4*hidden_size, 1]
            w_times_h_prev = w_times_h_prev.squeeze(-1)  # [batch_size, 4*hidden_size]

            # 分别计算输入门(i)、遗忘门(f)、cell(g)、输出门(o)
            # it
            # i_t = torch.sigmoid(w_times_x[:, :hidden_size] + w_times_h_prev[:, :hidden_size] + b_ih[:hidden_size] + b_hh[:hidden_size])
            Wi_t = w_times_x[:, :hidden_size] + w_times_h_prev[:, :hidden_size]
            Wi_t_expanded = Wi_t.unsqueeze(1)


            Wi_t = self.mambaBlock1(Wi_t_expanded).squeeze(1)


            i_t = torch.sigmoid(Wi_t + b_ih[:hidden_size] + b_hh[:hidden_size])

            # ft
            # f_t = torch.sigmoid(
            #     w_times_x[:, hidden_size:2 * hidden_size] + w_times_h_prev[:, hidden_size:2 * hidden_size]
            #     + b_ih[hidden_size:2 * hidden_size] + b_hh[hidden_size:2 * hidden_size])
            Wf_t = w_times_x[:, hidden_size:2 * hidden_size] + w_times_h_prev[:, hidden_size:2 * hidden_size]
            f_t = torch.sigmoid(Wf_t + b_ih[hidden_size:2 * hidden_size] + b_hh[hidden_size:2 * hidden_size])

            # C-hat
            # g_t = torch.tanh(
            #     w_times_x[:, 2 * hidden_size:3 * hidden_size] + w_times_h_prev[:, 2 * hidden_size:3 * hidden_size]
            #     + b_ih[2 * hidden_size:3 * hidden_size] + b_hh[2 * hidden_size:3 * hidden_size])
            new =w_times_x[:, 2 * hidden_size:3 * hidden_size] + w_times_h_prev[:, 2 * hidden_size:3 * hidden_size]
            new_expanded = new.unsqueeze(1)
            new_expanded = self.mambaBlock1(new_expanded).squeeze(1)
            g_t= torch.tanh(new_expanded + b_ih[2 * hidden_size:3 * hidden_size] + b_hh[2 * hidden_size:3 * hidden_size])

            # Ot
            # o_t = torch.sigmoid(
            #     w_times_x[:, 3 * hidden_size:4 * hidden_size] + w_times_h_prev[:, 3 * hidden_size:4 * hidden_size]
            #     + b_ih[3 * hidden_size:4 * hidden_size] + b_hh[3 * hidden_size:4 * hidden_size])
            WO_t = w_times_x[:, 3 * hidden_size:4 * hidden_size] + w_times_h_prev[:, 3 * hidden_size:4 * hidden_size]
            o_t = torch.sigmoid(WO_t + b_ih[3 * hidden_size:4 * hidden_size] + b_hh[3 * hidden_size:4 * hidden_size])

            # Ct
            prev_c = f_t * prev_c + i_t * g_t
            # ht
            prev_h = o_t * torch.tanh(prev_c)

            output[:, t, :] = prev_h
            output = output.to(self.device)

        return output, (prev_h, prev_c)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):

        # x  (batch_size, seq_length, input_dim)
        batch_size, T, input_size = x.shape[0], x.shape[1], x.shape[2]
        h_0_fw = torch.zeros(batch_size, self.hidden_dim).to(self.device)  # 
        c_0_fw = torch.zeros(batch_size, self.hidden_dim).to(self.device)  # 
        h_0_bw = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        c_0_bw = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        # print('111',h_0_fw.device,c_0_fw.device,h_0_bw.device,c_0_bw.device)

        # output_custom = x
        out = x
        for layer in range(self.num_layers):
            output_custom = out

            w_ih_fw = getattr(self.lstm_layer, f'weight_ih_l{layer}')  # .to(self.device)
            w_hh_fw = getattr(self.lstm_layer, f'weight_hh_l{layer}')  # .to(self.device)
            b_ih_fw = getattr(self.lstm_layer, f'bias_ih_l{layer}')  # .to(self.device)
            b_hh_fw = getattr(self.lstm_layer, f'bias_hh_l{layer}')  # .to(self.device)

            w_ih_bw = getattr(self.lstm_layer, f'weight_ih_l{layer}_reverse')  # .to(self.device)
            w_hh_bw = getattr(self.lstm_layer, f'weight_hh_l{layer}_reverse')  # .to(self.device)
            b_ih_bw = getattr(self.lstm_layer, f'bias_ih_l{layer}_reverse')  # .to(self.device)
            b_hh_bw = getattr(self.lstm_layer, f'bias_hh_l{layer}_reverse')  # .to(self.device)
            output_custom_b = out
            x_reversed = torch.flip(output_custom_b, [1])  # .to(self.device)  # 

            # output_custom=output_custom.to(self.device)
            output_custom_fw, (h_0_fw, c_0_fw) = self.lstm_forward(
                input=output_custom,
                initial_states=(h_0_fw, c_0_fw),
                w_ih=w_ih_fw,
                w_hh=w_hh_fw,
                b_ih=b_ih_fw,
                b_hh=b_hh_fw
            )

            output_custom_bw, (h_0_bw, c_0_bw) = self.lstm_forward(
                input=x_reversed,
                initial_states=(h_0_bw, c_0_bw),
                w_ih=w_ih_bw,
                w_hh=w_hh_bw,
                b_ih=b_ih_bw,
                b_hh=b_hh_bw
            )

            output_custom_b = torch.flip(output_custom_bw, [1])  # 
            out = torch.cat((output_custom_fw, output_custom_b),
                            dim=-1)  # out  (batch_size, seq_length, 2 * hidden_dim)
            if layer < self.num_layers - 1:
                out = self.dropout(out)

        return out

 

def main():
    msms_file='msms.txt'
    lbmsms_file='msms.txt'
    folder_path = '/home/inputData/inhouse/SampleSet'
    dfRT = pd.read_csv('./merged_predictions.csv', sep='\t', low_memory=False)  # sample
 
    full_path2 = os.path.join(folder_path, msms_file)
    folder_path2 = '/home/inputData/inhouse/LibrarySet'

    full_path3 = os.path.join(folder_path2, lbmsms_file)

    msms = pd.read_csv(full_path2, sep='\t', low_memory=False)  # sample
    lbmsms = pd.read_csv(full_path3, sep='\t', low_memory=False)  # lib
    mcb = MambaConvBiLSTM()
    mcb.fit(lbmsms) 
 

    dfSP = mcb.predict(dfRT, msms) 
    dfSP.to_csv('./Mamba_pred.csv')



    print('----------')


if __name__ == '__main__':
    main()

