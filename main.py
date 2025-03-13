import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from two_state import *
from trajectory_classes import *


NUM_DATASETS = 10000  # Increasing this value will improve RMSE estimate
TRAJECTORIES_PER_DATASET = 50
DOCTOR_COST_PER_ANNOTATION = 10
LLM_COST_PER_ANNOTATION = 1  # Keep this at 1

def main():
  # Model
  state_distribution = np.array([0.5, 0.5])
  true_reward_means = np.array([
      [1., 1.5],
      [0., 0.],
  ])
  true_reward_stds = np.array([
      [0.5, 0.5],
      [0.5, 0.5],
  ])

  # Doctor Noise
  doctor_bias = np.array([
      [0.15, 0.15],
      [0.15, 0.15],
  ])
  doctor_std = np.array([
      [0.9, 0.9],
      [0.9, 0.9],
  ])

  # LLM Noise
  llm_bias = np.array([
      [0.3, 0.3],
      [0.3, 0.3],
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
      [0.4, 0.6],
      [0.5,  0.5],
  ])

  # Try out different budgeting strategies
  all_budgets_per_dataset = [0, 10, 20, 30, 40, 50]
  doctor_percent_spends = [0, 20, 40, 60, 80, 100]

  # After processing each dataset, store the value estimates here:
  # Baselines:
  IS_estimates = []
  # Non-baselines (not sure what to call em)
  ISplus_estimates = {
      budget: {percent: [] for percent in doctor_percent_spends} \
      for budget in all_budgets_per_dataset}
  DMplus_IS_estimates = {
      budget: {percent: [] for percent in doctor_percent_spends} \
      for budget in all_budgets_per_dataset}

  for _ in tqdm(range(NUM_DATASETS)):
    factual_dataset = generate_dataset_of_trajectories(
        state_distribution, true_reward_means, true_reward_stds,
        behavior_policy, num_trajectories=TRAJECTORIES_PER_DATASET)

    # Estimate the evaluation_policy's value with standard IS.
    IS_estimates.append(run_vanilla_IS(
        evaluation_policy, behavior_policy, factual_dataset))

    for budget_per_dataset in all_budgets_per_dataset:
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

        # Use each estimator to estimate a value for the evaluation policy.
        # Intentionally do not add in the true_reward_stds.
        doctor_annotations = generate_annotations(
            factual_dataset, num_doctor_annotations_per_dataset,
            true_reward_means + doctor_bias, doctor_std)
        llm_annotations = generate_annotations(
            factual_dataset, num_llm_annotations_per_dataset,
            true_reward_means + llm_bias, llm_std)

        # IS+
        ISplus_estimates[budget_per_dataset][doctor_percent_spend].append(
            run_ISplus(evaluation_policy, behavior_policy, factual_dataset,
                       [doctor_annotations, llm_annotations]))

        # DM+-IS
        DMplus_IS_estimates[budget_per_dataset][doctor_percent_spend].append(
            run_DMplus_IS(evaluation_policy, behavior_policy, factual_dataset,
                       [doctor_annotations, llm_annotations]))
          

  true_evaluation_policy_value = calculate_true_policy_value(
      evaluation_policy, state_distribution, true_reward_means)

  # IS+
  ISplus_rmses = {percent: [] for percent in doctor_percent_spends}
  for budget in all_budgets_per_dataset:
    for doctor_percent in doctor_percent_spends:
      ISplus_rmses[doctor_percent].append(calculate_policy_value_rmse(
          ISplus_estimates[budget][doctor_percent],
          true_evaluation_policy_value))

  # Calculate DM+ - IS RMSEs
  DMplus_IS_rmses = {percent: [] for percent in doctor_percent_spends}
  for budget in all_budgets_per_dataset:
    for doctor_percent in doctor_percent_spends:
      DMplus_IS_rmses[doctor_percent].append(calculate_policy_value_rmse(
          DMplus_IS_estimates[budget][doctor_percent],
          true_evaluation_policy_value))

  # IS+ plot
  for doctor_percent in doctor_percent_spends:
    plt.plot(all_budgets_per_dataset, ISplus_rmses[doctor_percent],
             label=f"IS+ @ {doctor_percent}%.")
  # IS baseline
  IS_rmse = calculate_policy_value_rmse(IS_estimates,
                                        true_evaluation_policy_value)
  plt.axhline(
      y=np.mean(IS_rmse), label="Ordinary IS (No annotations)", color="gray",
      linestyle="--")
  plt.xlabel("Total Budget Per Dataset")
  plt.ylabel("RMSE")
  plt.legend()
  plt.show()
  
  # DM+-IS plot
  for doctor_percent in doctor_percent_spends:
    plt.plot(all_budgets_per_dataset, DMplus_IS_rmses[doctor_percent],
             label=f"DM+-IS @ {doctor_percent}%.")
  # DM-IS baseline (using the shortcut of 0 budget to avoid separately creating
  # an unannotated estimator)
  plt.axhline(
      y=np.mean(DMplus_IS_rmses[0][0]), label="DM-IS (No annotations)",
      color="gray", linestyle="--")
  plt.xlabel("Total Budget Per Dataset")
  plt.ylabel("RMSE")
  plt.legend()
  plt.show()


if __name__ == "__main__":
  main()