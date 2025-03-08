import numpy as np

from trajectory_classes import *

RNG = np.random.default_rng(234)


def calculate_policy_value_rmse(
      estimated_policy_values: np.ndarray,
      true_policy_value: int,
) -> int:
  """
  Given the estimated_policy_values generated for one particular policy,
  and the true_policy_value of that policy, calculate the RMSE.

  One policy value should be estimated per dataset. Each dataset should have
  some number of trajectories.
  """
  return np.sqrt(np.mean((estimated_policy_values - true_policy_value) ** 2))


# TODO: Implement if needed
# Tang & Wiens distinguish a policy's value and it's value function; that's why
# we have this second function here. This calculates a single value for the
# policy, while the function above gives a value for each state for the given
# policy.
def calculate_true_policy_value(policy, state_distribution, reward_means):
  return np.sum(state_distribution @ (policy * reward_means))


# STATUS: Lightly tested
def generate_dataset_of_trajectories(
    state_distribution: np.ndarray,
    reward_means: np.ndarray,
    reward_stds: np.ndarray,
    policy: np.ndarray,
    # NOTE: In theory, bandits should always have only one timestep. In the
    # original code, they provided some flexibility to having multiple
    # timesteps, so we do the same in our implementation.
    trajectory_len: int = 1,
    num_trajectories: int = 1000,
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
    num_trajectories:  Number of trajectories to generate

  CREDIT: This function is based on a subsection of single_exp_setting() from
  https://github.com/MLD3/CounterfactualAnnot-SemiOPE/blob/main/synthetic/bandit_compare-2state.ipynb
  """
  # Not hard-coding these values to 2, in case we want to use this code in a
  # more general library
  num_states, num_actions = policy.shape

  trajectories = []

  for _ in range(num_trajectories):
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
    factual_dataset: list[Trajectory],
    num_annotations: int,
    annotated_reward_means: np.ndarray,
    annotated_reward_stds: np.ndarray,
) -> list[np.ndarray]:
  """
  Args:
    trajectories: A list of batch_size equal-length Trajectories.
    num_annotations: The total number of annotations to generate. Replaces `Pc'
        in the original code.
    annotated_reward_means: Shape (num_states, num_actions)
    annotated_reward_stds: Shape (num_states, num_actions)

  Return Value:
    all_annotations: A list of counterfactual annotations, one for each
        trajectory. Effective shape (batch_size, trajectory_len, num_actions)

  If bias and/or variance is to be added to the annotations, it should be done
  *before* the reward_[means|stds] are passed to this function.

  This function should be called several times in order to generate annotations
  of varying fidelities.

  CREDIT: This function is built upon a subsection of single_exp_setting() from
  https://github.com/MLD3/CounterfactualAnnot-SemiOPE/blob/main/synthetic/bandit_compare-2state.ipynb
  """
  # NOTE: In the original code, they determine the probability of obtaining a
  # counterfactual annotation based on the actual action observed (i.e.
  # Pc[xi,ai]). This is fine for the 2-state scenario, but we will want to
  # modify this behavior if we need to generalize this code to multi-action
  # scenarios.
  all_annotations = []

  num_timesteps = len(factual_dataset[0])  # Assuming equal length trajectories

  # If we ever venture outside of the 2-state problem, we'll need to factor
  # in the number of actions.
  max_possible_annotations = num_timesteps * len(factual_dataset)
  num_annotations = np.minimum(num_annotations, max_possible_annotations)
  indices_to_annotate = set(
      RNG.choice(max_possible_annotations, size=num_annotations, replace=False))

  total_timesteps_processed = 0
  for trajectory in factual_dataset:
    # Generate the annotations for all flagged timesteps
    counterfac_rewards = np.full((num_timesteps, 2), np.nan)
    for timestep, state, action, _ in trajectory:
      if total_timesteps_processed in indices_to_annotate:
        counterfac_rewards[timestep, 1 - action] = RNG.normal(
            annotated_reward_means[state, 1 - action],
            annotated_reward_stds[state, 1 - action])
      total_timesteps_processed += 1

    all_annotations.append(counterfac_rewards)

  return all_annotations


def create_combined_factual_rewards_and_annotations(
    factual_dataset: list[Trajectory],
    annotations: list[list[np.ndarray]],
) -> np.ndarray:
  """
  Args:
    factual_dataset: A list of batch_size Trajectories.
    annotations: Effective shape: (num_annotation_sets, batch_size,
        trajectory_len, num_actions)
  
  Output:
    combined_factual_rewards_and_annotations: np.ndarray of shape
        (1 + num_annotation_sets, batch_size, trajectory_len, num_actions).
        np.nan will be found wherever counterfactual observations are not
        observed.
  """
  # Our annotations have np.nan where counterfactual observations are not
  # observed. Mimic this structure for the true rewards, and stack the rewards.
  stacked_nan_expanded_factual_rewards = np.array(
      [traj.create_nan_expanded_rewards() for traj in factual_dataset])
  stacked_nan_expanded_factual_rewards = np.expand_dims(
      stacked_nan_expanded_factual_rewards, 0)

  # Combine the rewards and annotations into one np.ndarray.
  return np.concatenate(
      (stacked_nan_expanded_factual_rewards, np.array(annotations)),
      axis=0)


def create_combined_states(
    factual_dataset: list[Trajectory]
) -> np.ndarray:
  """
  Output: Shape (batch_size, trajectory_len)
  """
  return np.stack([trajectory.states for trajectory in factual_dataset])


# STATUS: Needs testing
def run_vanilla_IS(
    policy_e: np.ndarray,
    policy_b: np.ndarray,
    dataset: list[Trajectory],
) -> np.float32:
  """
  Run vanilla Importance Sampling to generate an estimate of the value of
  policy_e for the given dataset.

  Args:
    policy_e: The evaluation policy
    policy_b: The behavior policy
    dataset: A list of trajectories

  (The policy's value is not to be confused with the policy's value *function*)

  CREDIT: This function is based on a subsection of single_exp_setting() from
  https://github.com/MLD3/CounterfactualAnnot-SemiOPE/blob/main/synthetic/bandit_compare-2state.ipynb
  """
  ordinary_IS_value_estimates = []
  
  for trajectory in dataset:
    states, actions, rewards = trajectory.unpack()

    # Compute the inverse propensity scores for each timestep. Denoted as `rho'
    # in the literature.
    inv_prop_scores = policy_e[states, actions] / policy_b[states, actions]

    ordinary_IS_value_estimates.append(
        np.sum(inv_prop_scores * rewards) / len(trajectory))
    # TODO: Evaluate if we really do need to use weighted_IS_value_estimates.
    # weighted_IS_value_estimates.append(
    #     np.sum(inv_prop_scores * rewards) / np.sum(inv_prop_scores))
  
  return np.mean(ordinary_IS_value_estimates)


# This is the algorithm proposed by Tang & Wiens. Denoted as C-IS in their
# original paper, and as IS+ in Aishwarya's paper.
# STATUS: Needs implementation
def run_ISplus(
    policy_e: np.ndarray,
    policy_b: np.ndarray,
    dataset: list[Trajectory],
    annotations: list[list[np.ndarray]]
) -> np.ndarray:
  """
  This is a modified version of C-IS from Tang & Wiens (denoted IS+ by Mandyam
  et. al) that allows for multiple sets of annotations per trajectory.

  Args:
    dataset: A list of batch_size Trajectories.
    annotations: Effective shape: (num_annotation_sets, batch_size,
        trajectory_len, num_actions)

  """
  combined_factual_rewards_and_annotations = (
      create_combined_factual_rewards_and_annotations(dataset, annotations))

  ordinary_ISplus_value_estimates = []
  for trajectory in dataset:
    # Calculate the Inverse Propensity Scores (`rho' in the literature) for ALL
    # possible actions, given each observed state.
    # This is probably inefficient, but it doesn't matter for 2-state.
    inv_prop_scores = policy_e[trajectory.states] / policy_b[trajectory.states]
    # (trajectory_len, num_actions) --> (1, 1, trajectory_len, num_actions)
    inv_prop_scores = inv_prop_scores[np.newaxis, np.newaxis, :]

    # TODO: We may want to allow for a weighted mean here between factual
    # rewards and CFAs.
    ordinary_ISplus_value_estimates.append(
      np.nanmean(combined_factual_rewards_and_annotations * inv_prop_scores))
  
  return np.mean(ordinary_ISplus_value_estimates)


# Performed the best in Aishwarya's CANDOR paper
# STATUS: Needs implementation
def run_DMplus_IS(
    policy_e: np.ndarray,
    policy_b: np.ndarray,
    dataset: list[Trajectory],
    annotations: list[list[np.ndarray]],
):
  # Shape: (1 + num_annotation_sets, batch_size, trajectory_len, num_actions)
  combined_factual_rewards_and_annotations = (
      create_combined_factual_rewards_and_annotations(dataset, annotations))

  # Shape: (batch_size, trajectory_len)
  combined_states = create_combined_states(dataset)
  # Shape: (1, batch_size, trajectory_len, 1)
  combined_states = np.expand_dims(combined_states, (0, 3))

  # Estimate one reward function given all trajectories
  num_states = dataset[0].num_possible_states
  num_actions = dataset[0].num_possible_actions
  estimated_reward_func = np.zeros((num_states, num_actions))
  for state in range(num_states):
    # TODO: this may not be the cleanest
    state_mask = np.where(combined_states == state, 1, np.nan)

    # (May be a potential source of error later, if trying to nanmean without
    # having any non-nan values)
    estimated_reward_func[state] = np.nanmean(
        state_mask * combined_factual_rewards_and_annotations,
        axis=3)  # One value for each action

  value_estimates = []
  for trajectory in dataset:
    factual_states, factual_actions, factual_rewards = trajectory.unpack()

    inv_prop_scores = policy_e[factual_states] / policy_b[factual_states]
    # (trajectory_len, num_actions) --> (1, 1, trajectory_len, num_actions)
    inv_prop_scores = inv_prop_scores[np.newaxis, np.newaxis, :]

    estimated_policy_e_rewards = np.sum(
        estimated_reward_func[factual_states] * policy_e[factual_states],
        axis=1)
    estimated_factual_rewards = estimated_reward_func[factual_states,
                                                      factual_actions]

    value_estimates.append(np.mean(
        estimated_policy_e_rewards \
        + inv_prop_scores * (factual_rewards - estimated_factual_rewards)))

# TODO: We may come up with a new algorithm here for evaluating a policy. If
# that is the case, then we can treat the above evaluation algorithms as
# baselines.
