import numpy as np

from two_state import *
from trajectory_classes import *


# TODO: Run IS without any annotations.
# TODO: Run IS+ and DM+-IS in two scenarios:
# - With *only* high quality annotations
# - With *both* high and low quality annotations


# ==============================================================================
# JF: Below is scratch code used to verify the behaviors of the functions in
#     two_state.py.
# ==============================================================================

def verify_IS_behavior():
  state_distribution = np.array([1, 0])  # Only ever start in state 0.
  # Second action is more optimal in state 0.
  reward_means = np.array([
      [1., 2.],  
      [0., 0.],
  ])
  reward_stds = np.array([
      [0.5, 0.5],
      [0.5, 0.5],
  ])

  # Create a scenario where the evaluation policy should perform better.
  behavior_policy = np.array([
      [0.5, 0.5],
      [0.5, 0.5],
  ])
  evaluation_policy = np.array([
      [0.25, 0.75],
      [0.5,  0.5],
  ])

  # Only generate trajectories for the behavior policy
  trajectories = generate_dataset_of_trajectories(state_distribution,
                                  reward_means,
                                  reward_stds,
                                  behavior_policy)

  IS_estimate = run_vanilla_IS(
      evaluation_policy, behavior_policy, trajectories)
  
  print("Should be exactly 1.75: ",
        calculate_true_policy_value(evaluation_policy, state_distribution,
                                    reward_means))
  
  # 1 * 0.25 + 2 * 0.75 = 1.75
  print("Should be roughly equal to 1.75: ", IS_estimate)

if __name__ == "__main__":
  verify_IS_behavior()