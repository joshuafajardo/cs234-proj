import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from two_state import *
from trajectory_classes import *

# Constants for faster execution
NUM_DATASETS = 100  # Reduced for faster execution
DOCTOR_COST_PER_ANNOTATION = 20
LLM_COST_PER_ANNOTATION = 1
FIXED_BUDGET = 200

def sample_size_analysis():
    """
    Analyze the effect of sample size on RMSE.
    """
    print("Analyzing effect of sample size...")
    
    # Environment parameters
    state_distribution = np.array([0.5, 0.5])
    true_reward_means = np.array([
        [1.0, 2.0],  
        [0.5, 0.0],
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
        [0.25, 0.75],
        [0.5,  0.5],
    ])
    
    # Annotation parameters
    doctor_bias = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    doctor_std = np.array([
        [0.8, 0.8],
        [0.8, 0.8],
    ])
    llm_bias = np.array([
        [0.2, 0.2],
        [0.2, 0.2],
    ])
    llm_std = np.array([
        [0.6, 0.6],
        [0.6, 0.6],
    ])
    
    # Fixed budget with reasonable allocation
    doctor_percent_spend = 15  # Based on optimal for DM+-IS
    doctor_spend = int((doctor_percent_spend / 100) * FIXED_BUDGET 
                    // DOCTOR_COST_PER_ANNOTATION 
                    * DOCTOR_COST_PER_ANNOTATION)
    llm_spend = int((FIXED_BUDGET - doctor_spend) 
                    // LLM_COST_PER_ANNOTATION 
                    * LLM_COST_PER_ANNOTATION)
    
    num_doctor_annotations = doctor_spend // DOCTOR_COST_PER_ANNOTATION
    num_llm_annotations = llm_spend // LLM_COST_PER_ANNOTATION
    
    # True policy value
    true_evaluation_policy_value = calculate_true_policy_value(
        evaluation_policy, state_distribution, true_reward_means)
    
    # Sample sizes to test
    sample_sizes = [10, 25, 50, 100, 200]
    
    # Store results
    vanilla_IS_rmses = []
    ISplus_rmses = []
    DMplus_IS_rmses = []
    
    for trajectories_per_dataset in tqdm(sample_sizes):
        # Initialize estimates
        vanilla_IS_estimates = []
        ISplus_estimates = []
        DMplus_IS_estimates = []
        
        for _ in range(NUM_DATASETS):
            # Generate dataset
            factual_dataset = generate_dataset_of_trajectories(
                state_distribution, true_reward_means, true_reward_stds,
                behavior_policy, num_trajectories=trajectories_per_dataset)
            
            # Vanilla IS
            vanilla_IS_estimates.append(run_vanilla_IS(
                evaluation_policy, behavior_policy, factual_dataset))
            
            # Generate annotations
            doctor_annotations = generate_annotations(
                factual_dataset, num_doctor_annotations,
                true_reward_means + doctor_bias, doctor_std)
            llm_annotations = generate_annotations(
                factual_dataset, num_llm_annotations,
                true_reward_means + llm_bias, llm_std)
            
            # Run OPE algorithms
            ISplus_estimates.append(
                run_ISplus(evaluation_policy, behavior_policy, factual_dataset,
                       [doctor_annotations, llm_annotations]))
            DMplus_IS_estimates.append(
                run_DMplus_IS(evaluation_policy, behavior_policy, factual_dataset,
                       [doctor_annotations, llm_annotations]))
        
        # Calculate RMSE
        vanilla_IS_rmses.append(calculate_policy_value_rmse(
            vanilla_IS_estimates, true_evaluation_policy_value))
        ISplus_rmses.append(calculate_policy_value_rmse(
            ISplus_estimates, true_evaluation_policy_value))
        DMplus_IS_rmses.append(calculate_policy_value_rmse(
            DMplus_IS_estimates, true_evaluation_policy_value))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, vanilla_IS_rmses, 'o-', label='Vanilla IS')
    plt.plot(sample_sizes, ISplus_rmses, 's-', label='IS+')
    plt.plot(sample_sizes, DMplus_IS_rmses, '^-', label='DM+-IS')
    
    plt.title('Effect of Sample Size on RMSE')
    plt.xlabel('Number of Trajectories per Dataset')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('sample_size_vs_rmse.png')
    plt.close()
    
    # Calculate and plot relative improvement
    rel_improvement_isplus = [(vanilla - isplus) / vanilla * 100 
                             for vanilla, isplus in zip(vanilla_IS_rmses, ISplus_rmses)]
    rel_improvement_dmplus = [(vanilla - dmplus) / vanilla * 100 
                             for vanilla, dmplus in zip(vanilla_IS_rmses, DMplus_IS_rmses)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, rel_improvement_isplus, 's-', label='IS+ Improvement')
    plt.plot(sample_sizes, rel_improvement_dmplus, '^-', label='DM+-IS Improvement')
    
    plt.title('Relative Improvement Over Vanilla IS by Sample Size')
    plt.xlabel('Number of Trajectories per Dataset')
    plt.ylabel('Improvement (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('sample_size_vs_improvement.png')
    plt.close()
    
    return {
        'sample_sizes': sample_sizes,
        'vanilla_IS_rmses': vanilla_IS_rmses,
        'ISplus_rmses': ISplus_rmses,
        'DMplus_IS_rmses': DMplus_IS_rmses
    }

def bias_variance_heatmap():
    """
    Create a 2D heatmap showing RMSE for different combinations of bias and variance.
    """
    print("Creating bias-variance heatmap...")
    
    # Environment parameters
    state_distribution = np.array([0.5, 0.5])
    true_reward_means = np.array([
        [1.0, 2.0],  
        [0.5, 0.0],
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
        [0.25, 0.75],
        [0.5,  0.5],
    ])
    
    # Fixed expert parameters
    doctor_bias = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    doctor_std = np.array([
        [0.8, 0.8],
        [0.8, 0.8],
    ])
    
    # Bias and variance values to test
    bias_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    std_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Fixed budget and allocation
    doctor_percent_spend = 15  # Based on optimal for DM+-IS
    doctor_spend = int((doctor_percent_spend / 100) * FIXED_BUDGET 
                    // DOCTOR_COST_PER_ANNOTATION 
                    * DOCTOR_COST_PER_ANNOTATION)
    llm_spend = int((FIXED_BUDGET - doctor_spend) 
                    // LLM_COST_PER_ANNOTATION 
                    * LLM_COST_PER_ANNOTATION)
    
    num_doctor_annotations = doctor_spend // DOCTOR_COST_PER_ANNOTATION
    num_llm_annotations = llm_spend // LLM_COST_PER_ANNOTATION
    
    # True policy value
    true_evaluation_policy_value = calculate_true_policy_value(
        evaluation_policy, state_distribution, true_reward_means)
    
    # Initialize heatmap matrices
    ISplus_heatmap = np.zeros((len(bias_values), len(std_values)))
    DMplus_IS_heatmap = np.zeros((len(bias_values), len(std_values)))
    
    for i, bias in enumerate(tqdm(bias_values)):
        for j, std in enumerate(std_values):
            # Set LLM annotation parameters
            llm_bias = np.array([
                [bias, bias],
                [bias, bias],
            ])
            llm_std = np.array([
                [std, std],
                [std, std],
            ])
            
            # Initialize estimates
            ISplus_estimates = []
            DMplus_IS_estimates = []
            
            for _ in range(NUM_DATASETS // 2):  # Reduced for faster execution
                # Generate dataset
                factual_dataset = generate_dataset_of_trajectories(
                    state_distribution, true_reward_means, true_reward_stds,
                    behavior_policy, num_trajectories=50)
                
                # Generate annotations
                doctor_annotations = generate_annotations(
                    factual_dataset, num_doctor_annotations,
                    true_reward_means + doctor_bias, doctor_std)
                llm_annotations = generate_annotations(
                    factual_dataset, num_llm_annotations,
                    true_reward_means + llm_bias, llm_std)
                
                # Run OPE algorithms
                ISplus_estimates.append(
                    run_ISplus(evaluation_policy, behavior_policy, factual_dataset,
                           [doctor_annotations, llm_annotations]))
                DMplus_IS_estimates.append(
                    run_DMplus_IS(evaluation_policy, behavior_policy, factual_dataset,
                           [doctor_annotations, llm_annotations]))
            
            # Calculate RMSE
            ISplus_heatmap[i, j] = calculate_policy_value_rmse(
                ISplus_estimates, true_evaluation_policy_value)
            DMplus_IS_heatmap[i, j] = calculate_policy_value_rmse(
                DMplus_IS_estimates, true_evaluation_policy_value)
    
    # Plot heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # IS+ heatmap
    im1 = ax1.imshow(ISplus_heatmap, cmap='viridis_r', aspect='auto', 
               extent=[min(std_values), max(std_values), max(bias_values), min(bias_values)])
    ax1.set_title('IS+ RMSE by Bias and Variance')
    ax1.set_xlabel('LLM Standard Deviation')
    ax1.set_ylabel('LLM Bias')
    fig.colorbar(im1, ax=ax1, label='RMSE')
    
    # DM+-IS heatmap
    im2 = ax2.imshow(DMplus_IS_heatmap, cmap='viridis_r', aspect='auto',
               extent=[min(std_values), max(std_values), max(bias_values), min(bias_values)])
    ax2.set_title('DM+-IS RMSE by Bias and Variance')
    ax2.set_xlabel('LLM Standard Deviation')
    ax2.set_ylabel('LLM Bias')
    fig.colorbar(im2, ax=ax2, label='RMSE')
    
    plt.tight_layout()
    plt.savefig('bias_variance_heatmap.png')
    plt.close()
    
    # Plot the difference heatmap
    plt.figure(figsize=(8, 6))
    diff_heatmap = ISplus_heatmap - DMplus_IS_heatmap
    im3 = plt.imshow(diff_heatmap, cmap='RdBu', aspect='auto',
                extent=[min(std_values), max(std_values), max(bias_values), min(bias_values)])
    plt.colorbar(im3, label='IS+ RMSE - DM+-IS RMSE')
    plt.title('RMSE Difference (IS+ - DM+-IS)')
    plt.xlabel('LLM Standard Deviation')
    plt.ylabel('LLM Bias')
    plt.tight_layout()
    plt.savefig('bias_variance_difference.png')
    plt.close()
    
    return {
        'bias_values': bias_values,
        'std_values': std_values,
        'ISplus_heatmap': ISplus_heatmap,
        'DMplus_IS_heatmap': DMplus_IS_heatmap,
        'diff_heatmap': diff_heatmap
    }

def optimal_allocation_vs_cost_ratio():
    """
    Analyze how optimal budget allocation changes with different cost ratios.
    """
    print("Analyzing optimal allocation vs cost ratio...")
    
    # Environment parameters
    state_distribution = np.array([0.5, 0.5])
    true_reward_means = np.array([
        [1.0, 2.0],  
        [0.5, 0.0],
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
        [0.25, 0.75],
        [0.5,  0.5],
    ])
    
    # Annotation parameters
    doctor_bias = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    doctor_std = np.array([
        [0.8, 0.8],
        [0.8, 0.8],
    ])
    llm_bias = np.array([
        [0.2, 0.2],
        [0.2, 0.2],
    ])
    llm_std = np.array([
        [0.6, 0.6],
        [0.6, 0.6],
    ])
    
    # Cost ratios to test
    cost_ratios = [5, 10, 20, 50, 100]
    doctor_percent_spends = [0, 5, 10, 15, 20, 30, 50, 70, 100]
    
    # True policy value
    true_evaluation_policy_value = calculate_true_policy_value(
        evaluation_policy, state_distribution, true_reward_means)
    
    # Store results
    optimal_allocations = {'IS+': [], 'DM+-IS': []}
    all_rmses = {ratio: {'IS+': {}, 'DM+-IS': {}} for ratio in cost_ratios}
    
    for cost_ratio in tqdm(cost_ratios):
        # Set cost per annotation
        doctor_cost = cost_ratio  # Relative to LLM cost = 1
        llm_cost = 1
        
        ISplus_rmses = {}
        DMplus_IS_rmses = {}
        
        for doctor_percent in doctor_percent_spends:
            # Calculate budget expenditure
            doctor_spend = int((doctor_percent / 100) * FIXED_BUDGET)
            llm_spend = FIXED_BUDGET - doctor_spend
            
            # Calculate number of annotations
            num_doctor_annotations = doctor_spend // doctor_cost
            num_llm_annotations = llm_spend // llm_cost
            
            # Skip if no annotations are possible
            if num_doctor_annotations == 0 and num_llm_annotations == 0:
                continue
            
            # Initialize estimates
            ISplus_estimates = []
            DMplus_IS_estimates = []
            
            for _ in range(NUM_DATASETS // 2):  # Reduced for faster execution
                # Generate dataset
                factual_dataset = generate_dataset_of_trajectories(
                    state_distribution, true_reward_means, true_reward_stds,
                    behavior_policy, num_trajectories=50)
                
                # Generate annotations
                doctor_annotations = generate_annotations(
                    factual_dataset, num_doctor_annotations,
                    true_reward_means + doctor_bias, doctor_std)
                llm_annotations = generate_annotations(
                    factual_dataset, num_llm_annotations,
                    true_reward_means + llm_bias, llm_std)
                
                # Run OPE algorithms
                ISplus_estimates.append(
                    run_ISplus(evaluation_policy, behavior_policy, factual_dataset,
                           [doctor_annotations, llm_annotations]))
                DMplus_IS_estimates.append(
                    run_DMplus_IS(evaluation_policy, behavior_policy, factual_dataset,
                           [doctor_annotations, llm_annotations]))
            
            # Calculate RMSE
            ISplus_rmses[doctor_percent] = calculate_policy_value_rmse(
                ISplus_estimates, true_evaluation_policy_value)
            DMplus_IS_rmses[doctor_percent] = calculate_policy_value_rmse(
                DMplus_IS_estimates, true_evaluation_policy_value)
            
            # Store all RMSEs
            all_rmses[cost_ratio]['IS+'][doctor_percent] = ISplus_rmses[doctor_percent]
            all_rmses[cost_ratio]['DM+-IS'][doctor_percent] = DMplus_IS_rmses[doctor_percent]
        
        # Find optimal allocations
        if ISplus_rmses:
            optimal_isplus = min(ISplus_rmses.items(), key=lambda x: x[1])[0]
            optimal_allocations['IS+'].append(optimal_isplus)
        else:
            optimal_allocations['IS+'].append(None)
        
        if DMplus_IS_rmses:
            optimal_dmplus = min(DMplus_IS_rmses.items(), key=lambda x: x[1])[0]
            optimal_allocations['DM+-IS'].append(optimal_dmplus)
        else:
            optimal_allocations['DM+-IS'].append(None)
    
    # Plot optimal allocations vs cost ratio
    plt.figure(figsize=(10, 6))
    plt.plot(cost_ratios, optimal_allocations['IS+'], 'o-', label='IS+')
    plt.plot(cost_ratios, optimal_allocations['DM+-IS'], 's-', label='DM+-IS')
    
    plt.title('Optimal Expert Budget Allocation vs Cost Ratio')
    plt.xlabel('Expert:LLM Cost Ratio')
    plt.ylabel('Optimal % of Budget for Expert Annotations')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('optimal_allocation_vs_cost_ratio.png')
    plt.close()
    
    # Create RMSE vs allocation plots for each cost ratio
    for cost_ratio in cost_ratios:
        plt.figure(figsize=(10, 6))
        
        percentages = sorted(all_rmses[cost_ratio]['IS+'].keys())
        isplus_values = [all_rmses[cost_ratio]['IS+'][p] for p in percentages]
        dmplus_values = [all_rmses[cost_ratio]['DM+-IS'][p] for p in percentages]
        
        plt.plot(percentages, isplus_values, 'o-', label='IS+')
        plt.plot(percentages, dmplus_values, 's-', label='DM+-IS')
        
        plt.title(f'RMSE vs Expert Budget Allocation (Cost Ratio = {cost_ratio}:1)')
        plt.xlabel('Percentage of Budget for Expert Annotations')
        plt.ylabel('RMSE')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(f'rmse_vs_allocation_ratio_{cost_ratio}.png')
        plt.close()
    
    return {
        'cost_ratios': cost_ratios,
        'optimal_allocations': optimal_allocations,
        'all_rmses': all_rmses
    }

if __name__ == "__main__":
    # Run all analyses
    print("Running additional visualization analyses...")
    
    print("\n1. Sample Size Analysis")
    sample_size_results = sample_size_analysis()
    
    print("\n2. Bias-Variance Heatmap")
    bias_variance_results = bias_variance_heatmap()
    
    print("\n3. Optimal Allocation vs Cost Ratio")
    cost_ratio_results = optimal_allocation_vs_cost_ratio()
    
    print("\nAll analyses complete! Check the generated PNG files for visualizations.")