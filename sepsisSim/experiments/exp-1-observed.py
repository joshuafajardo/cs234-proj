# ## Simulation parameters
exp_name = 'exp-FINAL-1'
eps = 0.10
eps_str = '0_1'

run_idx_length = 1_000
N_val = 1_000
runs = 50


out_fname = f'./results/{exp_name}/vaso_eps_{eps_str}-observed.csv'

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pickle
import itertools
import copy
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn import metrics
import itertools

import joblib
from joblib import Parallel, delayed

from OPE_utils_new import (
    format_data_tensor,
    policy_eval_analytic_finite,
    OPE_IS_h,
    compute_behavior_policy_h,
)


def policy_eval_helper(π):
    V_H = policy_eval_analytic_finite(P.transpose((1,0,2)), R, π, gamma, H)
    Q_H = [(R + gamma * P.transpose((1,0,2)) @ V_H[h]) for h in range(1,H)] + [R]
    J = isd @ V_H[0]
    # Check recursive relationships
    assert len(Q_H) == H
    assert len(V_H) == H
    assert np.all(Q_H[-1] == R)
    assert np.all(np.sum(π * Q_H[-1], axis=1) == V_H[-1])
    assert np.all(R + gamma * P.transpose((1,0,2)) @ V_H[-1] == Q_H[-2])
    return V_H, Q_H, J

def iqm(x):
    return scipy.stats.trim_mean(x, proportiontocut=0.25, axis=None)

NSTEPS = H = 20   # max episode length in historical data # Horizon of the MDP
G_min = -1        # the minimum possible return
G_max =  1        # the maximum possible return
nS, nA = 1442, 8

PROB_DIAB = 0.2

# Ground truth MDP model
MDP_parameters = joblib.load('../data/MDP_parameters.joblib')
P = MDP_parameters['transition_matrix_absorbing'] # (A, S, S_next)
R = MDP_parameters['reward_matrix_absorbing_SA'] # (S, A)
nS, nA = R.shape
gamma = 0.99

# unif rand isd, mixture of diabetic state
isd = joblib.load('../data/modified_prior_initial_state_absorbing.joblib')
isd = (isd > 0).astype(float)
isd[:720] = isd[:720] / isd[:720].sum() * (1-PROB_DIAB)
isd[720:] = isd[720:] / isd[720:].sum() * (PROB_DIAB)

# Precomputed optimal policy
π_star = joblib.load('../data/π_star.joblib')



# ## Load data
input_dir = f'../datagen/vaso_eps_{eps_str}-100k/'

def load_data(fname):
    print('Loading data', fname, '...', end='')
    df_data = pd.read_csv('{}/{}'.format(input_dir, fname)).rename(columns={'State_idx': 'State'})#[['pt_id', 'Time', 'State', 'Action', 'Reward']]

    # Assign next state
    df_data['NextState'] = [*df_data['State'].iloc[1:].values, -1]
    df_data.loc[df_data.groupby('pt_id')['Time'].idxmax(), 'NextState'] = -1
    df_data.loc[(df_data['Reward'] == -1), 'NextState'] = 1440
    df_data.loc[(df_data['Reward'] == 1), 'NextState'] = 1441

    assert ((df_data['Reward'] != 0) == (df_data['Action'] == -1)).all()

    print('DONE')
    return df_data

# df_seed1 = load_data('1-features.csv') # tr
df_seed2 = load_data('2-features.csv') # va


# ## Policies

# vaso unif, mv abx optimal
π_unif = (np.tile(π_star.reshape((-1,2,2,2)).sum(axis=3, keepdims=True), (1,1,1,2)).reshape((-1, 8)) / 2)


# ### Behavior policy

# vaso eps=0.5, mv abx optimal
π_beh = (np.tile(π_star.reshape((-1,2,2,2)).sum(axis=3, keepdims=True), (1,1,1,2)).reshape((-1, 8)) / 2)
π_beh[π_star == 1] = 1-eps
π_beh[π_beh == 0.5] = eps

V_H_beh, Q_H_beh, J_beh = policy_eval_helper(π_beh)
J_beh


# ### Optimal policy
V_H_star, Q_H_star, J_star = policy_eval_helper(π_star)
J_star



# ## Compare OPE

# ### observed behavior

df_results_0 = []
for run in range(runs):
    df_va = df_seed2.set_index('pt_id').loc[200000+run*run_idx_length:200000+run*run_idx_length + N_val - 1].reset_index()[
        ['pt_id', 'Time', 'State', 'Action', 'Reward', 'NextState']
    ]
    df = df_va[['pt_id', 'Time', 'State', 'Action', 'Reward', 'NextState']]

    # OPE - WIS/WDR prep
    data_va = format_data_tensor(df)
    pi_b_va = compute_behavior_policy_h(df)

    # OPE - IS
    IS_value, WIS_value, info = OPE_IS_h(data_va, pi_b_va, π_star, gamma, epsilon=0.0)
    df_results_0.append([info['G'].mean()])

df_results_0 = pd.DataFrame(df_results_0, columns=['IS_value'])
df_results_0.to_csv(out_fname, index=False)
