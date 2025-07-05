import numpy as np
import pandas as pd
import scipy as sp
import lightgbm as lgb
import networkx as nx
import matplotlib.pyplot as plt
import os

import torch.nn.functional as F
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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


class RetentionTimePredictor:
    def __init__(self, min_peptide_occurrence=3, random_state=42):
        """
        Parameters:
        -----------
        min_peptide_occurrence : int
            Minimum occurrence of peptides to be included in modeling
        random_state : int
            Random seed for reproducibility
        """
        self.min_occurrence = min_peptide_occurrence
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = ElasticNet(random_state=random_state)
        self.feature_names = None

    def _preprocess_data(self, df, file_col, peptide_col, rt_col):
        """Create unique identifier and filter low-frequency peptides"""
        df = df.copy()
        df['File_Pep'] = df[file_col] + '+' + df[peptide_col]

        # Filter peptides with sufficient occurrences
        peptide_counts = df[peptide_col].value_counts()
        valid_peptides = peptide_counts[peptide_counts >= self.min_occurrence].index
        print(len(valid_peptides))
        return df[df[peptide_col].isin(valid_peptides)]

    def _preprocess_data_pre(self, df, file_col, peptide_col, rt_col):
        """Create unique identifier and filter low-frequency peptides"""
        df = df.copy()
        df['File_Pep'] = df[file_col] + '+' + df[peptide_col]

        # Filter peptides with sufficient occurrences
        peptide_counts = df[peptide_col].value_counts()
        valid_peptides = peptide_counts[peptide_counts >= 1].index
        print(len(valid_peptides))

        return df[df[peptide_col].isin(valid_peptides)]
    def _create_rt_features(self, df, file_col, peptide_col, rt_col):
        """Generate retention time features for modeling"""
        # 聚合RT值，确保每个peptide+sample唯一

        df_agg = df.groupby([peptide_col, file_col])[rt_col].mean().reset_index()

        # 构建矩阵
        adj_matrix = df_agg.pivot(index=peptide_col, columns=file_col, values=rt_col)

        # 计算统计特征
        features = pd.DataFrame({
            'Median': adj_matrix.median(axis=1),
            'Gmean': adj_matrix.apply(lambda x: gmean(x.dropna()) if x.dropna().size > 0 else np.nan, axis=1),
            'Mean': adj_matrix.mean(axis=1),
            'Std': adj_matrix.std(axis=1),
            'CV': adj_matrix.std(axis=1) / adj_matrix.mean(axis=1) * 100,
            'Skew': adj_matrix.skew(axis=1)
        })

        print(features.shape)
        return features.dropna()

    def fit(self, df, file_col, peptide_col, rt_col, target_col):
        # Preprocess data (combine target and decoy)
        df_processed = self._preprocess_data(df, file_col, peptide_col, rt_col)

        # Generate features (using all peptides)
        features = self._create_rt_features(df_processed, file_col, peptide_col, rt_col)

        # Prepare training data
        X = self.scaler.fit_transform(features)
        y = df_processed.groupby(peptide_col)[rt_col].median().loc[features.index]

        # Store feature names for reference
        self.feature_names = features.columns.tolist()

        # Cross-validation to assess model performance
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        print(f"Cross-validated R² scores: {cv_scores}")
        print(f"Mean CV R²: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

        # Train final model
        self.model.fit(X, y)

        # Predictions on training data for evaluation
        y_pred = self.model.predict(X)


        # Compute other evaluation metrics
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)  # RMSE = sqrt(MSE)
        corr, _ = pearsonr(y, y_pred)

        print(f"\nEvaluation on full training set:")
        print(f"R²: {r2:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"Pearson Correlation: {corr:.6f}")

        return self

    def predict(self, df, file_col, peptide_col, rt_col=None):

        df_processed = self._preprocess_data_pre(df, file_col, peptide_col, rt_col) if rt_col else df
        features = self._create_rt_features(df_processed, file_col, peptide_col, rt_col)

        # Make predictions
        X = self.scaler.transform(features)
        pred_rt = self.model.predict(X)



        # Format results
        result = pd.DataFrame({
            'peptide': features.index,
            'predicted_RT': pred_rt
        })

        if rt_col:
            true_rt = df_processed.groupby(peptide_col)[rt_col].median().loc[features.index]
            result['deltaRT'] = np.log2(np.abs(true_rt.values - pred_rt) + 1)

        return result

    def evaluate_model(self, df, file_col, peptide_col, rt_col):
        """Evaluate model performance (overall R² for all peptides)"""

        # Get predictions
        predictions = self.predict(df, file_col, peptide_col, rt_col)

        # Merge原始RT和预测RT（按peptide）
        merged = pd.merge(
            df[[peptide_col, rt_col]].drop_duplicates(),
            predictions,
            left_on=peptide_col,
            right_on='peptide'
        )

        # 计算R²
        overall_r2 = r2_score(
            merged[rt_col],
            merged['predicted_RT']
        )

        print(f"Overall R²: {overall_r2:.3f}")

        return {
            'overall_r2': overall_r2
        }


# endregion

def main():

    folder_path = './inhouse/Maxquant/BulkSet/txt'

    full_path = os.path.join(folder_path, 'evidence.txt')
    evidence_lib = pd.read_csv(full_path, sep='\t', low_memory=False)  # sample

 
    folder_path2 = './dfSP_houseMamba_part1_pred.csv'
    evidence_samp = pd.read_csv(folder_path2, encoding="utf-8")  # sample

    rt_predictor = RetentionTimePredictor(min_peptide_occurrence=3)

    # 训练模型
    rt_predictor.fit(
        df=evidence_lib,
        file_col='Raw file',
        peptide_col='Modified sequence',
        rt_col='Retention time',
        target_col='label'
    )

    # 预测新数据
    predictions = rt_predictor.predict(
        df=evidence_samp,
        file_col='Raw file',
        peptide_col='Modified sequence',
        rt_col='Retention time'  # 可选，用于计算deltaRT
    )

  
    merged_data = pd.merge(
        evidence_samp,
        predictions,
        left_on="Modified sequence",
        right_on="peptide",
        how="left"
    )
    merged_data['PEPRT'] = merged_data['deltaRT'] * (1 + merged_data['PEP'])
    merged_data['scoreRT'] = merged_data['Score'] / (1 + merged_data['deltaRT'])
 
    merged_data.to_csv("./merged_predictions.csv", index=False)
 


    print('----------')


if __name__ == '__main__':

    t0 = time()
    main()
    print(' time: {} m {}s'.format(int((time() - t0) // 60), (time() - t0) % 60))
