import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from two_state import *
from trajectory_classes import *


NUM_DATASETS = 2000  # Increasing this value will improve RMSE estimate
TRAJECTORIES_PER_DATASET = 50
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
      [0., 0.],
      [0., 0.],
  ])
  doctor_std = np.array([
      [0.9, 0.9],
      [0.9, 0.9],
  ])

  # LLM Noise
  llm_bias = np.array([
      [0., 0.],
      [0., 0.],
  ])
  llm_std = np.array([
      [0.7, 0.7],
      [0.7, 0.7],
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
  # Generate a value for each dataset
  for _ in range(NUM_DATASETS):
    factual_dataset = generate_dataset_of_trajectories(
        state_distribution, true_reward_means, true_reward_stds,
        behavior_policy, num_trajectories=TRAJECTORIES_PER_DATASET)

    IS_estimates.append(run_vanilla_IS(
        evaluation_policy, behavior_policy, factual_dataset))

  # Try out different budgeting strategies
  all_budgets_per_dataset = [0, 100, 200, 300, 400, 500]
  doctor_percent_spends = [0, 10, 50, 90, 100]
  ISplus_estimates = {
      budget: {percent: [] for percent in doctor_percent_spends} \
      for budget in all_budgets_per_dataset}
  DMplus_IS_estimates = {
      budget: {percent: [] for percent in doctor_percent_spends} \
      for budget in all_budgets_per_dataset}

  for budget_per_dataset in tqdm(all_budgets_per_dataset):
    for doctor_percent_spend in doctor_percent_spends:
      # Calculate budgeted annotations
      doctor_spend = int((doctor_percent_spend / 100) \
          * budget_per_dataset \
          // DOCTOR_COST_PER_ANNOTATION \
          * DOCTOR_COST_PER_ANNOTATION)
      llm_spend = int((budget_per_dataset - doctor_spend) \
          // LLM_COST_PER_ANNOTATION \
          * LLM_COST_PER_ANNOTATION)

      num_doctor_annotations_per_dataset = doctor_spend \
          // DOCTOR_COST_PER_ANNOTATION
      num_llm_annotations_per_dataset = llm_spend // LLM_COST_PER_ANNOTATION

      # Generate a value for each dataset
      for _ in range(NUM_DATASETS):
        factual_dataset = generate_dataset_of_trajectories(
            state_distribution, true_reward_means, true_reward_stds,
            behavior_policy, num_trajectories=TRAJECTORIES_PER_DATASET)

        IS_estimates.append(run_vanilla_IS(
            evaluation_policy, behavior_policy, factual_dataset))
        
        # Intentionally do not add in the true_reward_stds.
        doctor_annotations = generate_annotations(
            factual_dataset, num_doctor_annotations_per_dataset,
            true_reward_means + doctor_bias, doctor_std)
        llm_annotations = generate_annotations(
            factual_dataset, num_llm_annotations_per_dataset,
            true_reward_means + llm_bias, llm_std)

        ISplus_estimates[budget_per_dataset][doctor_percent_spend].append(
            run_ISplus(evaluation_policy, behavior_policy, factual_dataset,
                       [doctor_annotations, llm_annotations]))
        DMplus_IS_estimates[budget_per_dataset][doctor_percent_spend].append(
            run_DMplus_IS(evaluation_policy, behavior_policy, factual_dataset,
                       [doctor_annotations, llm_annotations]))
          

  # Plotting
  true_evaluation_policy_value = calculate_true_policy_value(
      evaluation_policy, state_distribution, true_reward_means)

  # IS
  IS_rmse = calculate_policy_value_rmse(IS_estimates,
                                        true_evaluation_policy_value)
  plt.axhline(y=np.mean(IS_rmse), label="Ordinary IS")

  # IS+
  ISplus_rmses = {percent: [] for percent in doctor_percent_spends}
  for budget in all_budgets_per_dataset:
    for doctor_percent in doctor_percent_spends:
      ISplus_rmses[doctor_percent].append(calculate_policy_value_rmse(
          ISplus_estimates[budget][doctor_percent],
          true_evaluation_policy_value))

  # DM+ - IS
  DMplus_IS_rmses = {percent: [] for percent in doctor_percent_spends}
  for budget in all_budgets_per_dataset:
    for doctor_percent in doctor_percent_spends:
      DMplus_IS_rmses[doctor_percent].append(calculate_policy_value_rmse(
          DMplus_IS_estimates[budget][doctor_percent],
          true_evaluation_policy_value))

  for doctor_percent in doctor_percent_spends:
    plt.plot(all_budgets_per_dataset, ISplus_rmses[doctor_percent],
             label=f"IS+ @ {doctor_percent}%.")
    plt.plot(all_budgets_per_dataset, DMplus_IS_rmses[doctor_percent],
             label=f"DM+-IS @ {doctor_percent}%.")

  plt.title("RMSEs vs Total Budget for IS+ with Various Doctor-Budget "
            "Allocations (with IS baseline) in the Positive-Reward-Only "
            "Scenario.")
  plt.xlabel("Total Budget Per Dataset")
  plt.ylabel("RMSE")
  plt.legend()

  plt.show()
    

if __name__ == "__main__":
  main()