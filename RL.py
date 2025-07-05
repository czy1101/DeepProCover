import numpy as np
import pandas as pd
#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, auc, roc_curve
from copy import deepcopy
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

file_path = 'G:/czy/test/psm/log/trans/beifen/dfSP_nano_mamba_pred.csv'
data = pd.read_csv(file_path, encoding="utf-8")
# data2 = pd.read_csv("G:/czy/test/psm/log/trans/beifen/dfSP_houseMamba_part1_pred.csv", encoding="utf-8")
# data = pd.concat([data, data2], axis=0, ignore_index=True)

feature_columns = ['Length', 'Acetyl (Protein N-term)', 'Oxidation (M)', 'Missed cleavages',
                   'Charge', 'm/z', 'Mass', 'Mass error [ppm]', 'Retention length', 'PEP',
                   'MS/MS scan number', 'Score', 'Delta score', 'PIF', 'Intensity',
                   'Retention time',
                   'predicted_RT', 'deltaRT', 'PEPRT', 'scoreRT',
                   'Cosine', 'PEPCosine', 'ScoreCosine', 'CCS_prediction']
target_column = 'label'
file_column = 'Experiment'
protein_column = 'Leading razor protein'
X = deepcopy(data[feature_columns]).values
y = deepcopy(data[target_column]).values  # label
XX = deepcopy(data[feature_columns]).values

n_features = X.shape[1]


class PolicyNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.out = nn.Linear(128, input_dim)

    def forward(self, state):
        x = F.relu(self.norm1(self.fc1(state)))
        x = F.relu(self.norm2(self.fc2(x)))
        out = torch.sigmoid(self.out(x))
        return out


# ==== action ====
def select_features_static(policy_net, state, threshold=0.5):

    probs = policy_net(state)
    actions = (probs > threshold).float()
    return actions, probs

def select_features(policy_net, state, threshold=0.5):
    probs = policy_net(state) 
    dist = torch.distributions.Bernoulli(probs) 
    actions=dist.sample()
    #log_probs = dist.log_prob(actions).sum()  
    return actions,probs



# ====  LightGBM  reward ====
def evaluate_features(X, y, X_val, y_val, selected_features,clf):
    if selected_features.sum() == 0:
        return 0  
    selected = selected_features.bool().cpu().numpy()
    X_sub = X[:, selected]
    X_val_sub = X_val[:, selected]

    clf.fit(X_sub,y)
    pred = clf.predict_proba(X_val_sub)[:, 1]
    #scores = cross_val_score(clf, X_sub, y, cv=3, scoring='roc_auc')
    return roc_auc_score(y_val,pred)


# ==== RL ====
policy_net = PolicyNet(n_features)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
epochs = 300
baseline =0.0
alpha = 0.9

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
best_reward = -float('inf')
best_model_state = None
best_action = None

rewards_his = []
selected_counts = []

state = torch.ones(n_features)
for epoch in range(epochs):
    reward = 0
    actions, probs = select_features(policy_net, state)
    selected_counts.append(int(actions.sum()))

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        reward += evaluate_features(X_train, y_train, X_val, y_val, actions,model)


    reward /= skf.get_n_splits()
    rewards_his.append(reward)
    baseline = alpha*baseline+(1-alpha)*reward

    advantage = reward-baseline
    loss = -advantage * torch.sum(torch.log(probs + 1e-8) * actions.detach())

    # REINFORCE 
    if reward >best_reward:
        best_reward=reward
        best_model_state =policy_net.state_dict()
        best_action=actions

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    state= actions.detach()
    print(f"Epoch {epoch:03d}: Best_Reward = {reward:.4f}, Features selected = {int(actions.sum())}")
    print(actions)

df = pd.DataFrame({
    'reward': rewards_his,
    'selected_count': selected_counts
})

df.to_csv('./training_history.csv', index=False)


print(best_reward)

print(best_action)



