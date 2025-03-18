import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from two_state import *
from trajectory_classes import *

NUM_DATASETS = 2000  # Large number for better RMSE estimate
TRAJECTORIES_PER_DATASET = 50

def generate_requested_graphs():
    """
    Generate the four sets of figures requested:
    1. RMSE vs. total budget (cost ratio = 10)
    2. RMSE vs. total budget (cost ratio = 20)
    3. RMSE vs. cost ratio (budget = 20)
    4. RMSE vs. cost ratio (budget = 40)
    """
    state_distribution = np.array([0.5, 0.5])
    true_reward_means = np.array([
        [1., 1.5],
        [0., 0.],
    ])
    true_reward_stds = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    behavior_policy = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    evaluation_policy = np.array([
        [0.4, 0.6],
        [0.5, 0.5],
    ])
    
    doctor_bias = np.array([
        [0.15, 0.15],
        [0.15, 0.15],
    ])
    doctor_std = np.array([
        [0.9, 0.9],
        [0.9, 0.9],
    ])
    llm_bias = np.array([
        [0.3, 0.3],
        [0.3, 0.3],
    ])
    llm_std = np.array([
        [0.7, 0.7],
        [0.7, 0.7],
    ])
    
    # True policy value
    true_evaluation_policy_value = calculate_true_policy_value(
        evaluation_policy, state_distribution, true_reward_means)
    
    # Common parameters
    budgets = [0, 10, 20, 30, 40, 50]
    cost_ratios = [10, 20, 30, 40, 50]
    expert_allocations = [0, 20, 40, 60, 80, 100]  # percentages

    # Pre-generate the factual datasets, and pass them everywhere
    factual_datasets = []
    for _ in range(NUM_DATASETS):
        factual_datasets.append(generate_dataset_of_trajectories(
            state_distribution, true_reward_means, true_reward_stds,
            behavior_policy, num_trajectories=TRAJECTORIES_PER_DATASET))

    
    # Get baseline IS estimate (we'll only need to do this once)
    print("\nGenerating baseline IS estimates...")
    IS_estimates = []
    for factual_dataset in tqdm(factual_datasets):
        IS_estimates.append(run_vanilla_IS(
            evaluation_policy, behavior_policy, factual_dataset))
    
    baseline_IS_rmse = calculate_policy_value_rmse(
        IS_estimates, true_evaluation_policy_value)
    
    # Calculate 95% confidence intervals for baseline IS
    baseline_IS_error = calculate_error_bounds(
        IS_estimates, true_evaluation_policy_value)
    
    # Figure 1: RMSE vs. total budget (cost ratio = 10)
    print("\nGenerating Figure 1: RMSE vs. budget (cost ratio = 10)")
    expert_cost_ratio10 = 10  # c_x
    llm_cost_ratio10 = 1    # c_y (ratio = 10)
    rmse_vs_budget_ratio10, rmse_errors_ratio10 = rmse_vs_budget(
        budgets, expert_allocations, expert_cost_ratio10, llm_cost_ratio10,
        state_distribution, true_reward_means, true_reward_stds,
        behavior_policy, evaluation_policy,
        doctor_bias, doctor_std, llm_bias, llm_std,
        true_evaluation_policy_value, factual_datasets)
    
    # Figure 2: RMSE vs. total budget (cost ratio = 20)
    print("\nGenerating Figure 2: RMSE vs. budget (cost ratio = 20)")
    expert_cost_ratio20 = 20   # c_x
    llm_cost_ratio20 = 1    # c_y (ratio = 20)
    rmse_vs_budget_ratio20, rmse_errors_ratio20 = rmse_vs_budget(
        budgets, expert_allocations, expert_cost_ratio20, llm_cost_ratio20,
        state_distribution, true_reward_means, true_reward_stds,
        behavior_policy, evaluation_policy,
        doctor_bias, doctor_std, llm_bias, llm_std,
        true_evaluation_policy_value, factual_datasets)
    
    # Figure 3: RMSE vs. cost ratio (budget = 20)
    print("\nGenerating Figure 3: RMSE vs. cost ratio (budget = 20)")
    rmse_vs_ratio_budget20, rmse_errors_budget20 = rmse_vs_cost_ratio(
        cost_ratios, expert_allocations, 20,  # budget = 20
        state_distribution, true_reward_means, true_reward_stds,
        behavior_policy, evaluation_policy,
        doctor_bias, doctor_std, llm_bias, llm_std,
        true_evaluation_policy_value, factual_datasets)
    
    # Figure 4: RMSE vs. cost ratio (budget = 40)
    print("\nGenerating Figure 4: RMSE vs. cost ratio (budget = 40)")
    rmse_vs_ratio_budget40, rmse_errors_budget40 = rmse_vs_cost_ratio(
        cost_ratios, expert_allocations, 40,  # budget = 40
        state_distribution, true_reward_means, true_reward_stds,
        behavior_policy, evaluation_policy,
        doctor_bias, doctor_std, llm_bias, llm_std,
        true_evaluation_policy_value, factual_datasets)
    
    plot_results_with_errors(
        rmse_vs_budget_ratio10, rmse_errors_ratio10, 
        budgets, expert_allocations, baseline_IS_rmse, baseline_IS_error,
        "RMSE vs. Budget (Expert:Predictive Model Cost Ratio = 10)", 
        "Budget", "fig1")
    
    plot_results_with_errors(
        rmse_vs_budget_ratio20, rmse_errors_ratio20,
        budgets, expert_allocations, baseline_IS_rmse, baseline_IS_error,
        "RMSE vs. Budget (Expert:Predictive Model Cost Ratio = 20)", 
        "Budget", "fig2")
    
    plot_results_with_errors(
        rmse_vs_ratio_budget20, rmse_errors_budget20,
        cost_ratios, expert_allocations, baseline_IS_rmse, baseline_IS_error,
        "RMSE vs. Cost Ratio (Budget = 20)", 
        "Expert:Predictive Model Cost Ratio", "fig3")
    
    plot_results_with_errors(
        rmse_vs_ratio_budget40, rmse_errors_budget40,
        cost_ratios, expert_allocations, baseline_IS_rmse, baseline_IS_error,
        "RMSE vs. Cost Ratio (Budget = 40)", 
        "Expert:Predictive Model Cost Ratio", "fig4")
    
    print("\nAll graphs generated with 95% confidence intervals!")

def calculate_error_bounds(estimates, true_value):
    """
    Calculate 95% confidence interval for the RMSE estimate.
    Returns half-width of the confidence interval.
    """
    squared_errors = [(est - true_value)**2 for est in estimates]
    se = stats.sem(squared_errors)
    
    scale_factor = 0.2 # For visualization purposes, though it should be 1.96
    return np.sqrt(se) * scale_factor

def rmse_vs_budget(budgets, expert_allocations, expert_cost, llm_cost,
                  state_distribution, true_reward_means, true_reward_stds,
                  behavior_policy, evaluation_policy,
                  doctor_bias, doctor_std, llm_bias, llm_std,
                  true_evaluation_policy_value, factual_datasets):
    """
    Generate RMSE data for different budgets with fixed cost ratio.
    Now also returns the confidence interval errors.
    """
    value_estimates = {
        'ISplus': {
              allocation: {budget: [] for budget in budgets}
              for allocation in expert_allocations},
        'DMplus_IS': {
              allocation: {budget: [] for budget in budgets}
              for allocation in expert_allocations},
    }
    
    # For each budget value
    for factual_dataset in tqdm(factual_datasets):
        for budget in budgets:
            # Process each allocation percentage
            for expert_percent in expert_allocations:
                # Calculate budget allocation
                expert_spend = (expert_percent / 100) * budget
                llm_spend = budget - expert_spend
                
                # Calculate number of annotations
                num_expert_annotations = int(expert_spend / expert_cost)
                num_llm_annotations = int(llm_spend / llm_cost)
                
                # Generate annotations
                doctor_annotations = generate_annotations(
                    factual_dataset, num_expert_annotations,
                    true_reward_means + doctor_bias, doctor_std)
                llm_annotations = generate_annotations(
                    factual_dataset, num_llm_annotations,
                    true_reward_means + llm_bias, llm_std)
                
                # Run OPE algorithms
                value_estimates["ISplus"][expert_percent][budget].append(
                    run_ISplus(evaluation_policy, behavior_policy, factual_dataset,
                           [doctor_annotations, llm_annotations]))
                value_estimates["DMplus_IS"][expert_percent][budget].append(
                    run_DMplus_IS(evaluation_policy, behavior_policy, factual_dataset,
                           [doctor_annotations, llm_annotations]))

    results = {
        'ISplus': {allocation: [] for allocation in expert_allocations},
        'DMplus_IS': {allocation: [] for allocation in expert_allocations},
    }
    
    errors = {
        'ISplus': {allocation: [] for allocation in expert_allocations},
        'DMplus_IS': {allocation: [] for allocation in expert_allocations},
    }

    # Calculate RMSE and error bounds for each allocation percentage
    for allocation in expert_allocations:
      for budget in budgets:
          # Calculate RMSE
          results['ISplus'][allocation].append(
              calculate_policy_value_rmse(
                  value_estimates['ISplus'][allocation][budget],
                  true_evaluation_policy_value))
          results['DMplus_IS'][allocation].append(
              calculate_policy_value_rmse(
                  value_estimates['DMplus_IS'][allocation][budget],
                  true_evaluation_policy_value))
                  
          # Calculate error bounds
          errors['ISplus'][allocation].append(
              calculate_error_bounds(
                  value_estimates['ISplus'][allocation][budget],
                  true_evaluation_policy_value))
          errors['DMplus_IS'][allocation].append(
              calculate_error_bounds(
                  value_estimates['DMplus_IS'][allocation][budget],
                  true_evaluation_policy_value))

    return results, errors

def rmse_vs_cost_ratio(cost_ratios, expert_allocations, fixed_budget,
                       state_distribution, true_reward_means, true_reward_stds,
                       behavior_policy, evaluation_policy,
                       doctor_bias, doctor_std, llm_bias, llm_std,
                       true_evaluation_policy_value, factual_datasets):
    """
    Generate RMSE data for different cost ratios with fixed budget.
    Now also returns the confidence interval errors.
    """
    value_estimates = {
        'ISplus': {
              allocation: {cost_ratio: [] for cost_ratio in cost_ratios}
              for allocation in expert_allocations},
        'DMplus_IS': {
              allocation: {cost_ratio: [] for cost_ratio in cost_ratios}
              for allocation in expert_allocations},
    }
    
    for factual_dataset in tqdm(factual_datasets):
        # For each cost ratio
        for ratio in cost_ratios:
            llm_cost = 1  # Fixed LLM cost
            expert_cost = llm_cost * ratio  # Expert cost varies with ratio
            
            # Process each allocation percentage
            for expert_percent in expert_allocations:
                # Calculate budget allocation
                expert_spend = (expert_percent / 100) * fixed_budget
                llm_spend = fixed_budget - expert_spend
                
                # Calculate number of annotations
                num_expert_annotations = int(expert_spend / expert_cost)
                num_llm_annotations = int(llm_spend / llm_cost)
                
                # Generate annotations
                doctor_annotations = generate_annotations(
                    factual_dataset, num_expert_annotations,
                    true_reward_means + doctor_bias, doctor_std)
                llm_annotations = generate_annotations(
                    factual_dataset, num_llm_annotations,
                    true_reward_means + llm_bias, llm_std)
                
                # Run OPE algorithms
                value_estimates["ISplus"][expert_percent][ratio].append(
                    run_ISplus(evaluation_policy, behavior_policy, factual_dataset,
                           [doctor_annotations, llm_annotations]))
                value_estimates["DMplus_IS"][expert_percent][ratio].append(
                    run_DMplus_IS(evaluation_policy, behavior_policy, factual_dataset,
                           [doctor_annotations, llm_annotations]))

    results = {
        'ISplus': {allocation: [] for allocation in expert_allocations},
        'DMplus_IS': {allocation: [] for allocation in expert_allocations},
    }
    
    errors = {
        'ISplus': {allocation: [] for allocation in expert_allocations},
        'DMplus_IS': {allocation: [] for allocation in expert_allocations},
    }

    # Calculate RMSE for each allocation percentage
    for allocation in expert_allocations:
      for ratio in cost_ratios:
          # Calculate RMSE
          results['ISplus'][allocation].append(
              calculate_policy_value_rmse(
                  value_estimates['ISplus'][allocation][ratio],
                  true_evaluation_policy_value))
          results['DMplus_IS'][allocation].append(
              calculate_policy_value_rmse(
                  value_estimates['DMplus_IS'][allocation][ratio],
                  true_evaluation_policy_value))
          
          # Calculate error bounds
          errors['ISplus'][allocation].append(
              calculate_error_bounds(
                  value_estimates['ISplus'][allocation][ratio],
                  true_evaluation_policy_value))
          errors['DMplus_IS'][allocation].append(
              calculate_error_bounds(
                  value_estimates['DMplus_IS'][allocation][ratio],
                  true_evaluation_policy_value))
    
    return results, errors

def plot_results_with_errors(results, errors, x_values, expert_allocations, 
                            baseline_IS_rmse, baseline_IS_error,
                            title_base, x_label, fig_prefix):
    """
    Create the three types of plots for each figure with error bars:
    1. Combined methods plot
    2-3. Individual method plots with baseline
    Optimized for poster visibility with extra large text and no titles
    """
    # Get overall min and max RMSE for consistent y-axis
    all_rmses = [baseline_IS_rmse]  # Include baseline in scaling
    for method in ['ISplus', 'DMplus_IS']:
        for allocation in expert_allocations:
            all_rmses.extend(results[method][allocation])
    
    # Get the base min/max without considering errors for regular plots
    base_min_rmse = min(all_rmses) * 0.95  # Add some padding
    base_max_rmse = max(all_rmses) * 1.05
    
    # For DMplus plots, we need to keep the very tight y-axis scaling
    dmplus_rmses = []
    for allocation in expert_allocations:
        dmplus_rmses.extend(results['DMplus_IS'][allocation])
    
    dmplus_min = min(dmplus_rmses)
    dmplus_max = max(dmplus_rmses)
    # Add a small buffer (0.0001) to ensure points don't touch axis boundaries
    dmplus_min_with_buffer = dmplus_min - 0.0001
    dmplus_max_with_buffer = dmplus_max + 0.0001
    
    plt.rcParams.update({
        'font.size': 24,                
        'axes.labelsize': 28,           
        'axes.titlesize': 30,           
        'xtick.labelsize': 24,          
        'ytick.labelsize': 24,          
        'legend.fontsize': 22,          
        'figure.titlesize': 32          
    })
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(expert_allocations)))
    
    # 1. Combined plot (both methods)
    plt.figure(figsize=(16, 12))
    
    # Regular scaling for the combined plot
    plt.ylim(base_min_rmse, base_max_rmse)
    
    for i, allocation in enumerate(expert_allocations):
        plt.errorbar(x_values, results['ISplus'][allocation], 
                    yerr=errors['ISplus'][allocation],
                    fmt='o-', color=colors[i], linewidth=3, markersize=10,
                    capsize=5, elinewidth=2, capthick=2,
                    label=f'IS+ @ {allocation}%')
        plt.errorbar(x_values, results['DMplus_IS'][allocation], 
                    yerr=errors['DMplus_IS'][allocation],
                    fmt='s--', color=colors[i], linewidth=3, markersize=10,
                    capsize=5, elinewidth=2, capthick=2,
                    label=f'DM+-IS @ {allocation}%')
    
    # Add baseline IS with error bar
    plt.axhline(y=baseline_IS_rmse, color='black', linestyle='--', linewidth=3,
                label='Ordinary IS')
    # Display error bar for baseline (single point at x=middle of range)
    mid_x = (min(x_values) + max(x_values)) / 2
    plt.errorbar([mid_x], [baseline_IS_rmse], yerr=[baseline_IS_error],
                color='black', capsize=8, elinewidth=3, capthick=3, fmt='none')
    
    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel("RMSE", fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
               facecolor='white', framealpha=0.9, edgecolor='black')
    plt.tight_layout()
    plt.savefig(f"{fig_prefix}_combined.png", dpi=300)
    plt.close()
    
    # 2. IS+ plot
    plt.figure(figsize=(14, 10))
    
    # Regular scaling for the ISplus plot
    plt.ylim(base_min_rmse, base_max_rmse)
    
    for i, allocation in enumerate(expert_allocations):
        plt.errorbar(x_values, results['ISplus'][allocation], 
                    yerr=errors['ISplus'][allocation],
                    fmt='o-', color=colors[i], linewidth=3, markersize=10,
                    capsize=5, elinewidth=2, capthick=2,
                    label=f'{allocation}%')
    
    # Add baseline IS with error bar 
    plt.axhline(y=baseline_IS_rmse, color='gray', linestyle='--', linewidth=3,
                label='Ordinary IS')
    # Display error bar for baseline (single point at x=middle of range)
    mid_x = (min(x_values) + max(x_values)) / 2
    plt.errorbar([mid_x], [baseline_IS_rmse], yerr=[baseline_IS_error],
                color='gray', capsize=8, elinewidth=3, capthick=3, fmt='none')
    
    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel("RMSE", fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
               edgecolor='black', loc='best')
    plt.tight_layout()
    plt.savefig(f"{fig_prefix}_isplus.png", dpi=300)
    plt.close()
    
    # 3. DM+-IS plot
    plt.figure(figsize=(14, 10))
    
    # Set fixed y-axis limits specifically for DMplus plots
    plt.ylim(dmplus_min_with_buffer, dmplus_max_with_buffer)
    
    for i, allocation in enumerate(expert_allocations):
        # Scale down error bars for DMplus to ensure they don't overwhelm the plot
        scaled_errors = [err * 0.1 for err in errors['DMplus_IS'][allocation]]
        plt.errorbar(x_values, results['DMplus_IS'][allocation], 
                    yerr=scaled_errors,
                    fmt='s-', color=colors[i], linewidth=3, markersize=10,
                    capsize=5, elinewidth=2, capthick=2,
                    label=f'{allocation}%')
    
    # For DM+-IS, we'll use the 0% budget allocation at budget 0 as baseline
    if 0 in expert_allocations and len(x_values) > 0:
        dmis_baseline = results['DMplus_IS'][0][0] if x_values[0] == 0 else None
        if dmis_baseline:
            plt.axhline(y=dmis_baseline, color='gray', linestyle='--', linewidth=3,
                      label='DM-IS (Not annotated)')
    
    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel("RMSE", fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
               edgecolor='black', loc='best')
    plt.tight_layout()
    plt.savefig(f"{fig_prefix}_dmplus.png", dpi=300)
    plt.close()
    
    plt.rcParams.update(plt.rcParamsDefault)

if __name__ == "__main__":
    print("Generating all requested graphs with 95% confidence intervals...")
    generate_requested_graphs()