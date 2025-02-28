import numpy as np

from trajectory_classes import *

RNG = np.random.default_rng(234)


def calculate_policy_value_rmse(estimated_policy_values, true_policy_value):
  """
  Given the estimated_policy_values generated for one particular policy,
  and the true_policy_value of that policy, calculate the RMSE.
  """
  # TODO: Will need to call calculate_true_policy_value


# TODO: Implement if needed
# Tang & Wiens distinguish a policy's value and it's value function; that's why
# we have this second function here. This calculates a single value for the
# policy, while the function above gives a value for each state for the given
# policy.
def calculate_true_policy_value(policy, state_distribution, reward_means, reward_stds):
  pass


# STATUS: Needs testing
def generate_trajectories(
    state_distribution: np.ndarray,
    reward_means: np.ndarray,
    reward_stds: np.ndarray,
    policy: np.ndarray,
    # NOTE: In theory, bandits should always have only one timestep. In the
    # original code, they provided some flexibility to having multiple
    # timesteps, so we do the same in our implementation.
    trajectory_len: int = 1,
    num_runs: int = 1000,
):
  """
  Args:
    state_distribution:  Shape (num_states,). In the original code, this is
        d0. Since we're in a bandit setting, we can use this distribution for
        all timesteps.
    reward_means:  Shape (num_states, num_actions)
    reward_stds:  Shape (num_states, num_actions)
    policy:  Shape (num_states, num_actions)
    trajectory_len:  Trajectory length
    num_runs:  Number of trajectories to generate

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
        size=trajectory_len,
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
) -> list[np.ndarray]:
  """
  Args:
    trajectories: A list of batch_size equal-length Trajectories.
    counterfac_probs: Shape (num_states, num_actions). Named `Pc' in the
        original code.
    annotated_reward_means: Shape (num_states, num_actions)
    annotated_reward_stds: Shape (num_states, num_actions)

  Return Value:
    all_annotations: A list of counterfactual annotations, one for each
        trajectory. Effective shape (batch_size, num_actions, trajectory_length)

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
        [counterfac_probs[state, action] for _, state, action, _ in trajectory])
    counterfac_flags = RNG.random(num_timesteps) < curr_counterfac_probs

    # Generate the annotations for all flagged timesteps
    counterfac_rewards = np.full((2, num_timesteps), np.nan)
    for timestep, state, action, _ in trajectory:
      if counterfac_flags[timestep]:
        counterfac_rewards[1 - action, timestep] = RNG.normal(
            annotated_reward_means[state, 1 - action],
            annotated_reward_stds[state, 1 - action])

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

    ordinary_IS_value_estimates.append(
        np.sum(inv_prop_scores * rewards) / len(trajectory))
    weighted_IS_value_estimates.append(
        np.sum(inv_prop_scores * rewards) / np.sum(inv_prop_scores))
  
  return ordinary_IS_value_estimates, weighted_IS_value_estimates


# This is the algorithm proposed by Tang & Wiens. Denoted as C-IS in their
# original paper, and as IS+ in Aishwarya's paper.
# STATUS: Needs implementation
def run_ISplus(
    trajectories: list[Trajectory],
    annotations: list[list[np.ndarray]]
) -> np.ndarray:
  """
  This is a modified version of C-IS from Tang & Wiens (denoted IS+ by Mandyam
  et. al) that allows for multiple sets of annotations per trajectory.

  Args:
    trajectories: A list of batch_size Trajectories.
    annotations: Effective shape: (batch_size, num_annotation_sets, num_actions,
        trajectory_length)
  """
  # TODO: Remove the shortcut where we equally weight all annotations + factual
  # events.

  # Our annotations have np.nan where counterfactual observations are not
  # observed. Mimic this structure for the true rewards, and stack the rewards.
  stacked_nan_expanded_trajectories = np.array(
      [trajectory.create_nan_expanded_rewards() for trajectory in trajectories])
  stacked_nan_expanded_trajectories = np.expand_dims(
      stacked_nan_expanded_trajectories, 1)

  # Combine the rewards and annotations into one np.ndarray.
  # Shape: (batch_size, 1 + num_annotation_sets, num_actions, trajectory_length)
  combined_trajectories_and_annotations = np.concatenate(
      (stacked_nan_expanded_trajectories, np.array(annotations)),
      axis=1)

  


# Performed the best in Aishwarya's CANDOR paper
# STATUS: Needs implementation
def run_DMplus_IS():
  pass


# TODO: We may come up with a new algorithm here for evaluating a policy. If
# that is the case, then we can treat the above evaluation algorithms as
# baselines.
