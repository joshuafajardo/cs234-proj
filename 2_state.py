import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
%config InlineBackend.figure_formats = ['svg']
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.sans-serif'] = ['FreeSans']
import seaborn as sns
import itertools
from tqdm import tqdm
import joblib

from trajectories import *

RNG = np.random.default_rng(234)



def generate_trajectories(
    state_distribution: np.ndarray,
    reward_means: np.ndarray,
    reward_stds: np.ndarray,
    policy: np.ndarray,
    num_runs: int = 1000,
):
  """
  CREDIT: This function is based on a subsection of single_exp_setting() from
  https://github.com/MLD3/CounterfactualAnnot-SemiOPE/blob/main/synthetic/bandit_compare-2state.ipynb
  """
  # Not hard-coding these values to 2, in case we want to use this code in a
  # more general library
  num_states, num_actions = policy.shape

  trajectories = []

  for _ in range(num_runs):
    states = RNG.choice(
        num_states,
        size=num_runs,
        p=state_distribution)
    actions = np.array(
        [RNG.choice(num_actions, p=policy[state]) for state in states])
    rewards = np.array([
        RNG.normal(reward_means[state, action], reward_stds[state, action])
            for state, action in zip(states,actions)])

    trajectories.append(Trajectory(states, actions, rewards))
  
  return trajectories

def generate_annotations(
    counterfac_probs: np.ndarray,
    trajectories: list[Trajectory],
    annotated_reward_means: np.ndarray,
    annotated_reward_stds: np.ndarray,
):
  """
  If bias and/or variance is to be added to the annotations, it should be done
  *before* the reward_[means|stds] are passed to this function.

  This function should be called several times in order to generate annotations
  of varying fidelities.
  """
  # NOTE: In the original code, they determine the probability of obtaining a
  # counterfactual annotation based on the actual action observed (i.e.
  # Pc[xi,ai]). This is fine for the 2-state scenario, but we will want to
  # modify this behavior if we need to generalize this code to multi-action
  # scenarios.
  all_annotations = []

  for trajectory in trajectories:
    num_timesteps = len(trajectory)

    # Determine which timesteps should receive an annotation
    curr_counterfac_probs = np.array(
        [counterfac_probs[state, action] for state, action, _ in trajectory])
    counterfac_flags = RNG.random(num_timesteps) < curr_counterfac_probs

    # Generate the annotations for all timesteps (regardless of flag)
    counterfac_rewards = []
    for state, action, _ in trajectories:
      counterfac_rewards.append(RNG.normal(
          annotated_reward_means[state, 1 - action],
          annotated_reward_stds[state, 1 - action]))

    # Use the flags to mask timesteps that should receive no annotations.
    counterfac_rewards = np.where(counterfac_flags, counterfac_rewards, np.nan)
    all_annotations.append(counterfac_rewards)

  return all_annotations

def run_importance_sampling(pi_b, pi_e, reward_means, reward_stds, num_runs=1000):
  pass


# JF: Trying to break this function down into three separate functions above...
def single_exp_setting(pi_b, pi_e, reward_means, reward_stds, num_runs=1000):

  # True value of pi_e
  Js = []
  for seed in range(num_runs):
    rng = np.random.default_rng(seed=10+seed)
    x = rng.choice(len(d0), size=N, p=d0)
    a = np.array([rng.choice(2, p=pi_e[xi]) for xi in x])
    r = np.array([rng.normal(R[xi, ai], sigma[xi, ai]) for xi,ai in zip(x,a)])
    J = np.sum(r) / N
    Js.append(J)

  # Standard IS
  Gs = []
  OISs = []
  WISs = []
  for seed in range(num_runs):
    rng = np.random.default_rng(seed=10+seed)
    x = rng.choice(len(d0), size=N, p=d0)
    a = np.array([rng.choice(2, p=pi_b[xi]) for xi in x])
    r = np.array([rng.normal(R[xi,ai], sigma[xi,ai]) for xi,ai in zip(x,a)])
    G = np.sum(r) / N
    Gs.append(G)

    if use_piD:
      assert False
    else:
      pi_b_ = pi_b

    rho = pi_e[x,a] / pi_b_[x,a]
    OISs.append(np.sum(rho * r) / N)
    WISs.append(np.sum(rho * r) / np.sum(rho))

  
  # Collect data using pi_b - naive approach
  Naive_OISs = []
  for seed in range(num_runs):
    rng = np.random.default_rng(seed=10+seed)
    rng_c = np.random.default_rng(seed=100000+seed)
    x = rng.choice(len(d0), size=N, p=d0)
    a = np.array([rng.choice(2, p=pi_b[xi]) for xi in x])
    r = np.array([rng.normal(R[xi,ai], sigma[xi,ai]) for xi,ai in zip(x,a)])
    rho = pi_e[x,a] / pi_b[x,a]

    # counterfactual flag
    c = np.array([rng_c.choice(2, p=[1-Pc[xi,ai], Pc[xi,ai]]) for xi,ai in zip(x,a)])

    # counterfactual reward
    rc = np.array([rng_c.normal(R[xi,1-ai], sigma[xi,1-ai]) for xi,ai in zip(x,a)])
    rc[c==0] = np.nan

    # trajectory-wise weight
    w = np.ones(N)
    w[c==1] = ww_naive[x[c==1], a[c==1], a[c==1]]
    wc = np.zeros(N)
    wc[c==1] = ww_naive[x[c==1], a[c==1], 1-a[c==1]]

    if use_piD:
      # augmented behavior policy
      assert False
    else:
      # augmented behavior policy
      pi_b_ = np.array([
          [pi_b[0,0]*ww_naive[0,0,0]+pi_b[0,1]*ww_naive[0,1,0], pi_b[0,0]*ww_naive[0,0,1]+pi_b[0,1]*ww_naive[0,1,1]],
          [pi_b[1,0]*ww_naive[1,0,0]+pi_b[1,1]*ww_naive[1,1,0], pi_b[1,0]*ww_naive[1,0,1]+pi_b[1,1]*ww_naive[1,1,1]],
      ])
      pi_b_ = pi_b_ / pi_b_.sum(axis=1, keepdims=True)

    # Naive_WISs.append(
    #     (np.sum(w* pi_e[x,a] / pi_b_[x,a] * r) + np.nansum(wc* pi_e[x,1-a] / pi_b_[x,1-a] * rc)) / (np.sum(w* pi_e[x,a] / pi_b_[x,a]) + np.sum((wc* pi_e[x,1-a] / pi_b_[x,1-a])[c==1]))
    # )
    ## Add factual and counterfactual separately
    Naive_OISs.append(np.sum(pi_e[x,a] / pi_b_[x,a] * r) / N)
    if np.sum(c) > 0:
      Naive_OISs.append(np.sum(pi_e[x,1-a] / pi_b_[x,1-a] * rc) / np.sum(c))


  # Collect data using pi_b - combining counterfactuals with factuals
  FC_OISs_w = []
  FC_WISs_w = []
  for seed in range(num_runs):
    rng = np.random.default_rng(seed=10+seed)
    rng_c = np.random.default_rng(seed=100000+seed)
    x = rng.choice(len(d0), size=N, p=d0)
    a = np.array([rng.choice(2, p=pi_b[xi]) for xi in x])
    r = np.array([rng.normal(R[xi,ai], sigma[xi,ai]) for xi,ai in zip(x,a)])
    rho = pi_e[x,a] / pi_b[x,a]

    # counterfactual flag
    c = np.array([rng_c.choice(2, p=[1-Pc[xi,ai], Pc[xi,ai]]) for xi,ai in zip(x,a)])

    # counterfactual reward
    rc = np.array([rng_c.normal(R[xi,1-ai], sigma[xi,1-ai]) for xi,ai in zip(x,a)])
    rc[c==0] = np.nan

    # trajectory-wise weight
    w = np.ones(N)
    w[c==1] = ww[x[c==1], a[c==1], a[c==1]]
    wc = np.zeros(N)
    wc[c==1] = ww[x[c==1], a[c==1], 1-a[c==1]]

    if use_piD:
      # augmented behavior policy
      assert False
    else:
      # augmented behavior policy
      pi_b_ = np.array([
          [pi_b[0,0]*ww[0,0,0]+pi_b[0,1]*ww[0,1,0], pi_b[0,0]*ww[0,0,1]+pi_b[0,1]*ww[0,1,1]],
          [pi_b[1,0]*ww[1,0,0]+pi_b[1,1]*ww[1,1,0], pi_b[1,0]*ww[1,0,1]+pi_b[1,1]*ww[1,1,1]],
      ])
      pi_b_ = pi_b_ / pi_b_.sum(axis=1, keepdims=True)

    FC_OISs_w.append(
        (np.sum(w* pi_e[x,a] / pi_b_[x,a] * r) + np.nansum(wc* pi_e[x,1-a] / pi_b_[x,1-a] * rc)) / (N))
    FC_WISs_w.append(
        (np.sum(w* pi_e[x,a] / pi_b_[x,a] * r) + np.nansum(wc* pi_e[x,1-a] / pi_b_[x,1-a] * rc)) / (np.sum(w* pi_e[x,a] / pi_b_[x,a]) + np.sum((wc* pi_e[x,1-a] / pi_b_[x,1-a])[c==1])),)

  df_bias_var = []
  for name, values in [
      ('$\hat{v}(\pi_e)$', Js),
      ('$\hat{v}(\pi_b)$', Gs),
      ('OIS', OISs),
      ('WIS', WISs),
      ('C-OIS', FC_OISs_w),
      ('C-WIS', FC_WISs_w),
      ('Naive-OIS', Naive_OISs),
  ]:
    df_bias_var.append([name, 
                        np.mean(values), 
                        np.mean(values - d0@np.sum(pi_e*R, axis=1)), 
                        np.sqrt(np.var(values)), 
                        np.sqrt(np.mean(np.square(values - d0@np.sum(pi_e*R, axis=1))))])
  return pd.DataFrame(df_bias_var, columns=['Approach', 'Mean', 'Bias', 'Std', 'RMSE'])