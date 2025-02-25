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


# STATUS: Needs testing
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


# STATUS: Needs testing
def generate_annotations(
    trajectories: list[Trajectory],
    counterfac_probs: np.ndarray,
    annotated_reward_means: np.ndarray,
    annotated_reward_stds: np.ndarray,
):
  """
  If bias and/or variance is to be added to the annotations, it should be done
  *before* the reward_[means|stds] are passed to this function.

  This function should be called several times in order to generate annotations
  of varying fidelities.

  CREDIT: This function is based on a subsection of single_exp_setting() from
  https://github.com/MLD3/CounterfactualAnnot-SemiOPE/blob/main/synthetic/bandit_compare-2state.ipynb
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


# STATUS: Needs testing
def run_vanilla_IS(
    policy_e: np.ndarray,
    policy_b: np.ndarray,
    trajectories: list[Trajectory],
) -> tuple[np.ndarray, np.ndarray]:
  """
  Run vanilla Importance Sampling to generate an estimate of the value of
  policy_e for each trajectory.

  Args:
    policy_e: The evaluation policy
    policy_b: The behavior policy
    trajectories: A list of trajectories

  (The policy's value is not to be confused with the policy's value *function*)

  CREDIT: This function is based on a subsection of single_exp_setting() from
  https://github.com/MLD3/CounterfactualAnnot-SemiOPE/blob/main/synthetic/bandit_compare-2state.ipynb
  """
  ordinary_IS_value_estimates = []
  weighted_IS_value_estimates = []
  
  for trajectory in trajectories:
    states, actions, rewards = trajectory.unpack()

    # Compute the inverse propensity scores for each timestep. Denoted as `rho'
    # in the literature.
    inv_prop_scores = policy_e[states, actions] / policy_b[states, actions]

    ordinary_IS_value_estimates.append(np.sum(inv_prop_scores * rewards))
    weighted_IS_value_estimates.append(np.sum(inv_prop_scores * rewards) / np.sum(inv_prop_scores))
  
  return ordinary_IS_value_estimates, weighted_IS_value_estimates


# STATUS: Needs implementation
def run_ISplus():
  # AKA C-IS from Shengpu's paper
  pass


# STATUS: Needs implementation
def run_DM():
  pass


# Performed the best in Aishwarya's CANDOR paper
# STATUS: Needs implementation
def run_DMplus_IS():
  pass


# Not sure if we'll need this.
# STATUS: Needs implementation
def run_DM_ISplus():
  pass


# Not sure if we'll need this.
# STATUS: Needs implementation
def run_DMplus_ISplus():
  pass
