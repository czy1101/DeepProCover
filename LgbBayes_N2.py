# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @Desc     : DeepSCP: utilizing deep learning to boost single-cell proteome coverage


import numpy as np
import pandas as pd
import scipy as sp
import lightgbm as lgb
#import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns
#import torch.nn.functional as F
from scipy import stats

from copy import deepcopy
from time import time
from joblib import Parallel, delayed
from scipy.stats.mstats import gmean
from scipy.stats import pearsonr

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



warnings.filterwarnings('ignore')


# plt.rcParams['font.sans-serif'] = 'Arial'

# region method
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




# region DeepSpec 

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


class LGB_bayesianCV:
    def __init__(self, params_init=dict({}), n_splits=3, seed=0):
        self.n_splits = n_splits
        self.seed = seed
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'n_jobs': -1,
            'random_state': self.seed,
            'is_unbalance': True,
            'silent': True
        }
        self.params.update(params_init)

    def fit(self, x, y):
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        self.__x = np.array(x)
        self.__y = np.array(y)
        self.__lgb_bayesian()
        self.model = lgb.LGBMClassifier(**self.params)
        # self.cv_predprob = cross_val_predict(self.model, self.__x, self.__y,
        #                                      cv=self.skf,  method="predict_proba")[:, 1]
        self.model.fit(self.__x, self.__y)
        # self.feature_importance = dict(zip(self.model.feature_name_, self.model.feature_importances_))

    def predict(self, X):
        return self.model.predict(np.array(X))

    def predict_proba(self, X):
        return self.model.predict_proba(np.array(X))

    def __lgb_cv(self, n_estimators, learning_rate,
                 max_depth, num_leaves,
                 subsample, colsample_bytree,
                 min_split_gain, min_child_samples,
                 reg_alpha, reg_lambda):
        self.params.update({
            'n_estimators': int(n_estimators),  #  (100, 1000)
            'learning_rate': float(learning_rate),  #  (0.001, 0.3)
            'max_depth': int(max_depth),  #  (3, 15)
            'num_leaves': int(num_leaves),  # (2, 2^md) (5, 1000)
            'subsample': float(subsample),  #  (0.3, 0.9)
            'colsample_bytree': float(colsample_bytree),  #  (0.3, 0.9)
            'min_split_gain': float(min_split_gain),  #  (0, 0.5)
            'min_child_samples': int(min_child_samples),  # (5, 1000)
            'reg_alpha': float(reg_alpha),  #  (0, 10)
            'reg_lambda': float(reg_lambda),  #  (0, 10)
        })

        model = lgb.LGBMClassifier(**self.params)
        cv_score = cross_val_score(model, self.__x, self.__y, scoring="roc_auc", cv=self.skf).mean()
        return cv_score

    def __lgb_bayesian(self):
        lgb_bo = BayesianOptimization(self.__lgb_cv,
                                      {
                                          'n_estimators': (100, 1000),  #  (100, 1000)
                                          'learning_rate': (0.001, 0.3),  #  (0.001, 0.3)
                                          'max_depth': (3, 15),  #  (3, 15)
                                          'num_leaves': (5, 1000),  #  (2, 2^md) (5, 1000)
                                          'subsample': (0.3, 0.9),  #  (0.3, 0.9)
                                          'colsample_bytree': (0.3, 0.9),  # 
                                          'min_split_gain': (0, 0.5),  #  (0, 0.5)
                                          'min_child_samples': (5, 200),  # (5, 1000)
                                          'reg_alpha': (0, 10),  #  (0, 10)
                                          'reg_lambda': (0, 10),  # (0, 10)
                                      },
                                      random_state=self.seed,
                                      verbose=0)
        lgb_bo.maximize()
        self.best_auc = lgb_bo.max['target']
        lgbbo_params = lgb_bo.max['params']
        lgbbo_params['n_estimators'] = int(lgbbo_params['n_estimators'])
        lgbbo_params['learning_rate'] = float(lgbbo_params['learning_rate'])
        lgbbo_params['max_depth'] = int(lgbbo_params['max_depth'])
        lgbbo_params['num_leaves'] = int(lgbbo_params['num_leaves'])
        lgbbo_params['subsample'] = float(lgbbo_params['subsample'])
        lgbbo_params['colsample_bytree'] = float(lgbbo_params['colsample_bytree'])
        lgbbo_params['min_split_gain'] = float(lgbbo_params['min_split_gain'])
        lgbbo_params['min_child_samples'] = int(lgbbo_params['min_child_samples'])
        lgbbo_params['reg_alpha'] = float(lgbbo_params['reg_alpha'])
        lgbbo_params['reg_lambda'] = float(lgbbo_params['reg_lambda'])
        self.params.update(lgbbo_params)


class LgbBayes:
    def __init__(self, out_cv=3, inner_cv=3, seed=0):
        self.out_cv = out_cv
        self.inner_cv = inner_cv
        self.seed = seed

    def fit_tranform(self, data, feature_columns, target_column, file_column, protein_column=None):
        data_set = deepcopy(data)
        x = deepcopy(data_set[feature_columns]).values
        y = deepcopy(data_set[target_column]).values#label 
        skf = StratifiedKFold(n_splits=self.out_cv, shuffle=True, random_state=self.seed)
        cv_index = np.zeros(len(y), dtype=int)
        y_prob = np.zeros(len(y))
        y_pep = np.zeros(len(y))
        feature_importance_df = pd.DataFrame()

        for index, (train_index, test_index) in enumerate(skf.split(x, y)):
            print('++++++++++++++++CV {}+++++++++++++++'.format(index + 1))
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lgbbo = LGB_bayesianCV(n_splits=self.inner_cv)
            lgbbo.fit(x_train, y_train)

            y_testprob = lgbbo.predict_proba(x_test)[:, 1]
            y_prob[test_index] = y_testprob
            cv_index[test_index] = index + 1
            print('train auc:', lgbbo.best_auc)  # best val auc 或 lgbbo.model.best_score
            print('test auc:', roc_auc_score(y_test, y_testprob)) 
            y_pep[test_index] = Prob2PEP(y_test, y_testprob)

            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = feature_columns
            fold_importance_df["Importance"] = lgbbo.model.feature_importances_
            fold_importance_df["cv_index"] = index + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        y_qvalue = Score2Qval(y, -y_pep)# q-value 
        self.feature_imp = feature_importance_df[['Feature', 'Importance']].groupby(
            'Feature').mean().reset_index().sort_values(by='Importance')
        data_set['cv_index'] = cv_index
        data_set['Lgb_score'] = y_prob
        data_set['Lgb_pep'] = y_pep
        data_set['psm_qvalue'] = y_qvalue
        if protein_column is not None:
            data_set = GroupProteinPEP2Qval(data_set, file_column=file_column,
                                            protein_column=protein_column,
                                            target_column=target_column,
                                            pep_column='Lgb_pep')
        self.data_set = data_set
        return data_set

    def Feature_imp_plot(self):
        plt.figure(figsize=(6, 6))
        plt.barh(self.feature_imp.Feature, self.feature_imp.Importance, height=0.7, orientation="horizontal")
        plt.ylim(0, self.feature_imp.Feature.shape[0] - 0.35)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('Importance', fontsize=15)
        plt.ylabel('Features', fontsize=15)
        plt.tight_layout()

    def CVROC(self):
        index_column = 'cv_index'
        target_column = 'label'
        pred_column = 'Lgb_score'
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 10000)
        plt.figure(figsize=(5, 4))
        for i in sorted(self.data_set[index_column].unique()):
            y_true = self.data_set.loc[self.data_set[index_column] == i, target_column]
            y_prob = self.data_set.loc[self.data_set[index_column] == i, pred_column]

            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            mean_tpr += sp.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='Fold{} AUC = {}'.format(i, round(roc_auc, 4)))

        mean_tpr = mean_tpr / len(self.data_set[index_column].unique())
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, label='Mean AUC = {}'.format(round(mean_auc, 4)))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.legend(fontsize=12)
        plt.tight_layout()

    def PSM_accept(self):
        data = self.data_set[self.data_set['psm_qvalue'] <= 0.05]
        data = data.sort_values(by='psm_qvalue')
        data['Number of PSMs'] = range(1, data.shape[0] + 1)

        plt.figure(figsize=(5, 4))
        plt.plot(data['psm_qvalue'], data['Number of PSMs'], label='DeepProCover')
        plt.axvline(x=0.01, ls="--", c="gray")
        plt.xlim(-0.001, 0.05)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('PSM q-value', fontsize=15)
        plt.ylabel('Number of PSMs', fontsize=15)
        plt.legend(fontsize=13)
        plt.tight_layout()
# endregion


def PSM2ProPep(data, file_column,
               protein_column,
               peptide_column,
               intensity_columns):
    df = data[[file_column] + [protein_column] + [peptide_column] + intensity_columns]

    proteins = df.groupby([file_column, protein_column])[intensity_columns].sum(1).reset_index()
    df_pro = []
    for i, j in proteins.groupby([file_column]):
        k = pd.DataFrame(data=j[intensity_columns].values, index=j[protein_column].tolist(),
                         columns=['{}_{}'.format(i, x) for x in intensity_columns])
        df_pro.append(k)
    df_pro = pd.concat(df_pro, axis=1)
    df_pro[df_pro.isna()] = 0
    df_pro.index.name = protein_column

    peptides = df.groupby([file_column, protein_column, peptide_column])[intensity_columns].sum(1).reset_index()
    df_pep = []
    for i, j in peptides.groupby([file_column]):
        k = j.drop(file_column, axis=1).set_index([protein_column, peptide_column])
        k.columns = ['{}_{}'.format(i, x) for x in intensity_columns]
        df_pep.append(k)
    df_pep = pd.concat(df_pep, axis=1)
    df_pep[df_pep.isna()] = 0
    df_pep = df_pep.reset_index()
    return df_pro, df_pep


def proteinfilter(data, protein_count=15, sample_ratio=0.5):
    nrow = (data != 0).sum(1)
    data = data.loc[nrow[nrow >= protein_count].index]
    ncol = (data != 0).sum(0)
    data = data[ncol[ncol >= ncol.mean() * sample_ratio].index]
    return data



# endregion

def main():
    file_path = './N2_mamba_pred.csv'
    dfdb = pd.read_csv(file_path)

    feature_columns = ['Length', 'Acetyl (Protein N-term)',
                       'Oxidation (M)',
                       'Missed cleavages',
                       'Charge', 'm/z', 'Mass', 'Mass error [ppm]',
                        'Retention length',#0
                        'PEP',
                       'MS/MS scan number',
                       'Score',#0
                        'Delta score',
                        'PIF',#0
                        'Intensity',
                        'Retention time',

                       # 'predicted_RT',
                       # 'deltaRT',
                       # 'scoreRT',#0
                       # 'PEPRT',

                       #'CCS_prediction',
                       #'Cosine', 'PEPCosine', 'ScoreCosine'
                       ]

    target_column = 'label'
    file_column = 'Experiment'
    protein_column = 'Leading razor protein'
    lgs = LgbBayes()


    data_set = lgs.fit_tranform(data=dfdb,
                                feature_columns=feature_columns,
                                target_column=target_column,
                                file_column=file_column,
                                protein_column=protein_column)

    # region 
    feature_imp = lgs.feature_imp
    feature_imp.to_csv('./outputData/N2/feature_imp.csv', index=None)
    lgs.Feature_imp_plot()
    plt.savefig('./outputData/N2/figure/Feature_imp.pdf')
    lgs.CVROC()
    plt.savefig('./outputData/N2/figure/DeepSCP_ROC.pdf')
    lgs.PSM_accept()
    plt.savefig('./outputData/N2/figure/PSM_accept.pdf')
    # endregion

    data = data_set[(data_set.psm_qvalue < 0.01) & (data_set.protein_qvalue < 0.01) & (data_set.Lgb_pep < 0.01)&
                    (data_set.label == 1)]

    target=len(data)
    decoy =  data_set[(data_set.psm_qvalue < 0.01) & (data_set.protein_qvalue < 0.01) & (data_set.Lgb_pep < 0.01)&
                    (data_set.label == 0)]
    decoy2 = len(decoy)
    print(f"Target: {target}, Decoy: {decoy2}, Ratio: {target / (decoy2 + 1e-9):.2f}")

    peptide_column = 'Sequence'
    intensity_columns = [i for i in data.columns if 'Reporter intensity corrected' in i]

    df_pro, df_pep = PSM2ProPep(data, file_column=file_column,
                                protein_column=protein_column,
                                peptide_column=peptide_column,
                                intensity_columns=intensity_columns)

    data_set.to_csv('./outputData/N2/DeepSCP_evidence.txt', sep='\t', index=False)
    data.to_csv('./outputData/N2/DeepSCP_evidence_filter.txt', sep='\t', index=False)

    df_pro.to_csv('./outputData/N2/DeepSCP_pro.csv')
    df_pep.to_csv('./outputData/N2/DeepSCP_pep.csv', index=False)
    # endregion
    data.to_csv('./outputData/N2/data_.csv', index=None)
    data_set.to_csv('./outputData/N2/data_set_.csv', index=None)

    print('###################Protein filter###################')

    # region Type module
    an_cols = pd.DataFrame({'Sample_id': df_pro.columns,
                            'Set': [i.rsplit('_', 1)[0] for i in df_pro.columns],
                            'Channel': [i.rsplit('_', 1)[-1] for i in df_pro.columns]})
    an_cols['Cleanset'] = an_cols['Set'].str.strip("()'',")

    an_cols['Type'] = 'Empty'
    an_cols.loc[an_cols.Channel == 'Reporter intensity corrected 1', 'Type'] = 'Boost'
    an_cols.loc[an_cols.Channel == 'Reporter intensity corrected 2', 'Type'] = 'Reference'

    an_cols.loc[an_cols.Channel.isin(['Reporter intensity corrected 8',
                                      'Reporter intensity corrected 11',
                                      'Reporter intensity corrected 14']), 'Type'] = 'C10'
    an_cols.loc[an_cols.Channel.isin(['Reporter intensity corrected 9',
                                      'Reporter intensity corrected 12',
                                      'Reporter intensity corrected 15']), 'Type'] = 'RAW'
    an_cols.loc[an_cols.Channel.isin(['Reporter intensity corrected 10',
                                      'Reporter intensity corrected 13',
                                      'Reporter intensity corrected 16']), 'Type'] = 'SVEC'

    an_cols.to_csv('./outputData/N2/an_cols.csv', index=False)
    an_cols1 = an_cols[(an_cols.Type.isin(['C10', 'SVEC', 'RAW']))]
    an_cols1.Type.value_counts()
    df_pro1 = df_pro[list(set(df_pro.columns) & set(an_cols1.Sample_id))]

    # df_pro1 = df_pro[set(df_pro.columns) & set(an_cols1.Sample_id)]
    df_pep1 = df_pep[[protein_column] + [peptide_column] + df_pro1.columns.tolist()]

    df_pro_ft = proteinfilter(df_pro1)
    df_pep_ft = df_pep1[df_pep1[protein_column].isin(df_pro_ft.index)]
    (df_pro_ft != 0).sum(0).mean()
    (df_pep_ft.iloc[:, 2:] != 0).sum(0).mean()
    df_pro_ft.to_csv('./outputData/N2/pro_ft.csv')
    df_pep_ft.to_csv('./outputData/N2/pep_ft.csv')
    # endregion

    print('ok----------')



if __name__ == '__main__':

    t0 = time()
    main()
    print('DeepSCP using time: {} m {}s'.format(int((time() - t0) // 60), (time() - t0) % 60))
