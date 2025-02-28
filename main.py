import numpy as np

from two_state import *
from trajectory_classes import *


NUM_DATASETS = 100  # Increasing this value will only increase accuracy
TRAJECTORIES_PER_DATASET = 1000
ANNOTATION_BUDGET_PER_DATASET = 500
DOCTOR_COST_PER_ANNOTATION = 20
LLM_COST_PER_ANNOTATION = 1  # Keep this at 1

def main():
  # Model
  state_distribution = np.array([0.5, 0.5])
  true_reward_means = np.array([
      [1., 2.],  
      [0., 0.],
  ])
  true_reward_stds = np.array([
      [0.5, 0.5],
      [0.5, 0.5],
  ])

  # Doctor Noise
  doctor_bias = np.array([
      [0.25],
      [0.25],
  ])
  doctor_std = np.array([
      [0.5],
      [0.5],
  ])

  # LLM Noise
  llm_bias = np.array([
      [0.5],
      [0.5],
  ])
  llm_std = np.array([
      [0.25],
      [0.25],
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
  IS_estimates = []
  ISplus_estimates = []
  for num_doctor_annotations_per_dataset in [5]:  # TODO: expand this
    num_llm_annotations = ANNOTATION_BUDGET_PER_DATASET \
        - num_doctor_annotations_per_dataset * DOCTOR_COST_PER_ANNOTATION

    for _ in range(NUM_DATASETS):
      factual_dataset = generate_dataset_of_trajectories(
          state_distribution, true_reward_means, true_reward_stds,
          behavior_policy, num_runs=TRAJECTORIES_PER_DATASET)

      IS_estimates.append(run_vanilla_IS(
          evaluation_policy, behavior_policy, factual_dataset))
      
      doctor_annotations = generate_annotations(
          factual_dataset, num_doctor_annotations_per_dataset,
          true_reward_means + doctor_bias, true_reward_stds + doctor_std)
      llm_annotations = generate_annotations(
          factual_dataset, num_doctor_annotations_per_dataset,
          true_reward_means + llm_bias, true_reward_stds + llm_std)

      ISplus_estimates.append(run_ISplus(
          evaluation_policy, behavior_policy, factual_dataset,
          [doctor_annotations, doctor_bias]))
    

if __name__ == "__main__":
  main()