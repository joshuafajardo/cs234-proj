import numpy as np

import two_state
import trajectories


# TODO: Run IS without any annotations.
# TODO: Run IS+ and DM+-IS in two scenarios:
# - With *only* high quality annotations
# - With *both* high and low quality annotations


# ==============================================================================
# JF: Below is scratch code used to verify the behaviors of the functions in
#     two_state.py.
# ==============================================================================

def verify_IS_behavior():
  state_distribution = np.array([0.5, 0.5])
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
  trajectories = two_state.generate_trajectories(state_distribution,
                                  reward_means,
                                  reward_stds,
                                  behavior_policy)
  
  two_state.run_vanilla_IS(evaluation_policy, behavior_policy, trajectories)
  

if __name__ == "__main__":
  verify_IS_behavior()