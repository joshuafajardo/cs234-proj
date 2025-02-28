import numpy as np

from two_state import *
from trajectory_classes import *


NUM_DATASETS = 1  # Increasing this value will only increase accuracy. TODO increase this to 100
TRAJECTORIES_PER_DATASET = 1000
ANNOTATION_BUDGET_PER_DATASET = 500
DOCTOR_COST_PER_ANNOTATION = 20
LLM_COST_PER_ANNOTATION = 1  # Keep this at 1

def main():
  # Model
  state_distribution = np.array([1, 0])
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
      [0.1],
      [0.1],
  ])
  doctor_std = np.array([
      [0.6],
      [0.6],
  ])

  # LLM Noise
  llm_bias = np.array([
      [0.3],
      [0.3],
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
  # for doctor_percent_spend in [0, 10, 50, 90, 100]:
  for doctor_percent_spend in [50]:
    # Calculate budgeted annotations
    doctor_spend = int((doctor_percent_spend / 100) \
        * ANNOTATION_BUDGET_PER_DATASET \
        // DOCTOR_COST_PER_ANNOTATION \
        * DOCTOR_COST_PER_ANNOTATION)
    llm_spend = int((ANNOTATION_BUDGET_PER_DATASET - doctor_spend) \
        // LLM_COST_PER_ANNOTATION \
        * LLM_COST_PER_ANNOTATION)

    num_doctor_annotations_per_dataset = doctor_spend \
        // DOCTOR_COST_PER_ANNOTATION
    print(num_doctor_annotations_per_dataset)
    num_llm_annotations_per_dataset = llm_spend // LLM_COST_PER_ANNOTATION
    print(num_llm_annotations_per_dataset)

    # Generate a value for each dataset
    for _ in range(NUM_DATASETS):
      factual_dataset = generate_dataset_of_trajectories(
          state_distribution, true_reward_means, true_reward_stds,
          behavior_policy, num_trajectories=TRAJECTORIES_PER_DATASET)

      IS_estimates.append(run_vanilla_IS(
          evaluation_policy, behavior_policy, factual_dataset))
      
      doctor_annotations = generate_annotations(
          factual_dataset, num_doctor_annotations_per_dataset,
          true_reward_means + doctor_bias, true_reward_stds + doctor_std)
      llm_annotations = generate_annotations(
          factual_dataset, num_llm_annotations_per_dataset,
          true_reward_means + llm_bias, true_reward_stds + llm_std)

      ISplus_estimates.append(run_ISplus(
          evaluation_policy, behavior_policy, factual_dataset,
          [doctor_annotations, llm_annotations]))
    
    print("IS: ", np.mean(IS_estimates))
    print("IS+: ", np.mean(ISplus_estimates))
    

if __name__ == "__main__":
  main()