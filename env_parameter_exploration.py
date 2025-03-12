import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from two_state import *
from trajectory_classes import *

'''
DESCRIPTION: Tests different reward structures, 
annotation qualities, and expert budget allocations to determine optimal setting. 
'''

NUM_DATASETS = 200  
TRAJECTORIES_PER_DATASET = 50
DOCTOR_COST_PER_ANNOTATION = 20
LLM_COST_PER_ANNOTATION = 1

def explore_key_parameters():
    """
    Examining impact of 3 params: reward structure, annotation quality, expert budget allocations
    """
    state_distribution = np.array([0.5, 0.5])
    behavior_policy = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    evaluation_policy = np.array([
        [0.25, 0.75],
        [0.5,  0.5],
    ])
    
    # Test different reward structures
    reward_structures = [
        # Name, reward_means, reward_stds
        ("Base case", 
         np.array([[1.0, 2.0], [0.0, 0.0]]), 
         np.array([[0.5, 0.5], [0.5, 0.5]])),
        
        ("Higher rewards", 
         np.array([[2.0, 4.0], [0.0, 0.0]]), 
         np.array([[0.5, 0.5], [0.5, 0.5]])),
        
        ("Both states matter", 
         np.array([[1.0, 2.0], [0.5, 0.0]]), 
         np.array([[0.5, 0.5], [0.5, 0.5]])),
        
        ("Higher variance", 
         np.array([[1.0, 2.0], [0.0, 0.0]]), 
         np.array([[1.0, 1.0], [1.0, 1.0]]))
    ]
    
    # Test different annotation qualities
    annotation_qualities = [
        # Name, doctor_bias, doctor_std, llm_bias, llm_std
        ("Base case", 
         np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0.9, 0.9], [0.9, 0.9]]),
         np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0.7, 0.7], [0.7, 0.7]])),
        
        ("Biased LLM", 
         np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0.9, 0.9], [0.9, 0.9]]),
         np.array([[0.3, 0.3], [0.3, 0.3]]), np.array([[0.7, 0.7], [0.7, 0.7]])),
        
        ("Low variance LLM", 
         np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0.9, 0.9], [0.9, 0.9]]),
         np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0.4, 0.4], [0.4, 0.4]])),
        
        ("Realistic case", 
         np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[0.8, 0.8], [0.8, 0.8]]),
         np.array([[0.2, 0.2], [0.2, 0.2]]), np.array([[0.6, 0.6], [0.6, 0.6]]))
    ]
    
    budget = 200
    doctor_percent_spend = 10
    reward_results = {}
    annotation_results = {}
    
    # 1. Test different reward structures
    print("Testing different reward structures...")
    for name, reward_means, reward_stds in reward_structures:
        true_evaluation_policy_value = calculate_true_policy_value(
            evaluation_policy, state_distribution, reward_means)
        
        doctor_bias, doctor_std, llm_bias, llm_std = annotation_qualities[0][1:]
        
        doctor_spend = int((doctor_percent_spend / 100) * budget 
                          // DOCTOR_COST_PER_ANNOTATION 
                          * DOCTOR_COST_PER_ANNOTATION)
        llm_spend = int((budget - doctor_spend) 
                       // LLM_COST_PER_ANNOTATION 
                       * LLM_COST_PER_ANNOTATION)
        
        num_doctor_annotations = doctor_spend // DOCTOR_COST_PER_ANNOTATION
        num_llm_annotations = llm_spend // LLM_COST_PER_ANNOTATION
        
        vanilla_IS_estimates = []
        ISplus_estimates = []
        DMplus_IS_estimates = []
        
        for _ in range(NUM_DATASETS):
            factual_dataset = generate_dataset_of_trajectories(
                state_distribution, reward_means, reward_stds,
                behavior_policy, num_trajectories=TRAJECTORIES_PER_DATASET)
            
            vanilla_IS_estimates.append(run_vanilla_IS(
                evaluation_policy, behavior_policy, factual_dataset))
            
            doctor_annotations = generate_annotations(
                factual_dataset, num_doctor_annotations,
                reward_means + doctor_bias, doctor_std)
            llm_annotations = generate_annotations(
                factual_dataset, num_llm_annotations,
                reward_means + llm_bias, llm_std)
            
            ISplus_estimates.append(
                run_ISplus(evaluation_policy, behavior_policy, factual_dataset,
                       [doctor_annotations, llm_annotations]))
            DMplus_IS_estimates.append(
                run_DMplus_IS(evaluation_policy, behavior_policy, factual_dataset,
                       [doctor_annotations, llm_annotations]))
        
        vanilla_IS_rmse = calculate_policy_value_rmse(
            vanilla_IS_estimates, true_evaluation_policy_value)
        ISplus_rmse = calculate_policy_value_rmse(
            ISplus_estimates, true_evaluation_policy_value)
        DMplus_IS_rmse = calculate_policy_value_rmse(
            DMplus_IS_estimates, true_evaluation_policy_value)
        
        #improvement - measured by percentage
        isplus_improvement = (vanilla_IS_rmse - ISplus_rmse) / vanilla_IS_rmse * 100
        dmplus_improvement = (vanilla_IS_rmse - DMplus_IS_rmse) / vanilla_IS_rmse * 100
        
        reward_results[name] = {
            'vanilla_IS_rmse': vanilla_IS_rmse,
            'ISplus_rmse': ISplus_rmse,
            'DMplus_IS_rmse': DMplus_IS_rmse,
            'isplus_improvement': isplus_improvement,
            'dmplus_improvement': dmplus_improvement
        }
    
    # 2. Test different annotation qualities
    print("Testing different annotation qualities...")
    for name, doctor_bias, doctor_std, llm_bias, llm_std in annotation_qualities:
        reward_means, reward_stds = reward_structures[0][1:3]
        
        true_evaluation_policy_value = calculate_true_policy_value(
            evaluation_policy, state_distribution, reward_means)
        
        doctor_spend = int((doctor_percent_spend / 100) * budget 
                          // DOCTOR_COST_PER_ANNOTATION 
                          * DOCTOR_COST_PER_ANNOTATION)
        llm_spend = int((budget - doctor_spend) 
                       // LLM_COST_PER_ANNOTATION 
                       * LLM_COST_PER_ANNOTATION)
        
        num_doctor_annotations = doctor_spend // DOCTOR_COST_PER_ANNOTATION
        num_llm_annotations = llm_spend // LLM_COST_PER_ANNOTATION
        
        vanilla_IS_estimates = []
        ISplus_estimates = []
        DMplus_IS_estimates = []
        
        for _ in range(NUM_DATASETS):
            factual_dataset = generate_dataset_of_trajectories(
                state_distribution, reward_means, reward_stds,
                behavior_policy, num_trajectories=TRAJECTORIES_PER_DATASET)
            
            vanilla_IS_estimates.append(run_vanilla_IS(
                evaluation_policy, behavior_policy, factual_dataset))
            
            doctor_annotations = generate_annotations(
                factual_dataset, num_doctor_annotations,
                reward_means + doctor_bias, doctor_std)
            llm_annotations = generate_annotations(
                factual_dataset, num_llm_annotations,
                reward_means + llm_bias, llm_std)
            
            ISplus_estimates.append(
                run_ISplus(evaluation_policy, behavior_policy, factual_dataset,
                       [doctor_annotations, llm_annotations]))
            DMplus_IS_estimates.append(
                run_DMplus_IS(evaluation_policy, behavior_policy, factual_dataset,
                       [doctor_annotations, llm_annotations]))
        
        vanilla_IS_rmse = calculate_policy_value_rmse(
            vanilla_IS_estimates, true_evaluation_policy_value)
        ISplus_rmse = calculate_policy_value_rmse(
            ISplus_estimates, true_evaluation_policy_value)
        DMplus_IS_rmse = calculate_policy_value_rmse(
            DMplus_IS_estimates, true_evaluation_policy_value)
        
        #Improvement
        isplus_improvement = (vanilla_IS_rmse - ISplus_rmse) / vanilla_IS_rmse * 100
        dmplus_improvement = (vanilla_IS_rmse - DMplus_IS_rmse) / vanilla_IS_rmse * 100
        
        annotation_results[name] = {
            'vanilla_IS_rmse': vanilla_IS_rmse,
            'ISplus_rmse': ISplus_rmse,
            'DMplus_IS_rmse': DMplus_IS_rmse,
            'isplus_improvement': isplus_improvement,
            'dmplus_improvement': dmplus_improvement
        }
    
    # 3. Test optimal expert allocation
    print("Testing optimal expert allocation...")
    doctor_percentages = [0, 5, 10, 15, 20, 30, 50, 100]
    
    #Realistic case parameters
    name, doctor_bias, doctor_std, llm_bias, llm_std = annotation_qualities[3]
    reward_means, reward_stds = reward_structures[2][1:3]  #Both states matter
    
    true_evaluation_policy_value = calculate_true_policy_value(
        evaluation_policy, state_distribution, reward_means)
    
    allocation_results = {}
    
    for doctor_percent_spend in doctor_percentages:
        doctor_spend = int((doctor_percent_spend / 100) * budget 
                          // DOCTOR_COST_PER_ANNOTATION 
                          * DOCTOR_COST_PER_ANNOTATION)
        llm_spend = int((budget - doctor_spend) 
                       // LLM_COST_PER_ANNOTATION 
                       * LLM_COST_PER_ANNOTATION)
        
        num_doctor_annotations = doctor_spend // DOCTOR_COST_PER_ANNOTATION
        num_llm_annotations = llm_spend // LLM_COST_PER_ANNOTATION
        
        #Skip if no annotations are possible
        if num_doctor_annotations == 0 and num_llm_annotations == 0:
            continue
        
        ISplus_estimates = []
        DMplus_IS_estimates = []
        
        for _ in range(NUM_DATASETS):
            factual_dataset = generate_dataset_of_trajectories(
                state_distribution, reward_means, reward_stds,
                behavior_policy, num_trajectories=TRAJECTORIES_PER_DATASET)
            
            doctor_annotations = generate_annotations(
                factual_dataset, num_doctor_annotations,
                reward_means + doctor_bias, doctor_std)
            llm_annotations = generate_annotations(
                factual_dataset, num_llm_annotations,
                reward_means + llm_bias, llm_std)
            
            ISplus_estimates.append(
                run_ISplus(evaluation_policy, behavior_policy, factual_dataset,
                       [doctor_annotations, llm_annotations]))
            DMplus_IS_estimates.append(
                run_DMplus_IS(evaluation_policy, behavior_policy, factual_dataset,
                       [doctor_annotations, llm_annotations]))
        
        ISplus_rmse = calculate_policy_value_rmse(
            ISplus_estimates, true_evaluation_policy_value)
        DMplus_IS_rmse = calculate_policy_value_rmse(
            DMplus_IS_estimates, true_evaluation_policy_value)
        
        allocation_results[doctor_percent_spend] = {
            'ISplus_rmse': ISplus_rmse,
            'DMplus_IS_rmse': DMplus_IS_rmse
        }
    
    #Graph 1: Reward structure results
    plt.figure(figsize=(10, 6))
    scenarios = list(reward_results.keys())
    isplus_improvements = [reward_results[s]['isplus_improvement'] for s in scenarios]
    dmplus_improvements = [reward_results[s]['dmplus_improvement'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    plt.bar(x - width/2, isplus_improvements, width, label='IS+')
    plt.bar(x + width/2, dmplus_improvements, width, label='DM+-IS')
    
    plt.xlabel('Reward Structure')
    plt.ylabel('Improvement Over Vanilla IS (%)')
    plt.title('OPE Improvement by Reward Structure')
    plt.xticks(x, scenarios)
    plt.legend()
    
    plt.savefig('reward_structure_results.png')
    plt.close()
    
    #Graph 2: Annotation quality results
    plt.figure(figsize=(10, 6))
    scenarios = list(annotation_results.keys())
    isplus_improvements = [annotation_results[s]['isplus_improvement'] for s in scenarios]
    dmplus_improvements = [annotation_results[s]['dmplus_improvement'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    plt.bar(x - width/2, isplus_improvements, width, label='IS+')
    plt.bar(x + width/2, dmplus_improvements, width, label='DM+-IS')
    
    plt.xlabel('Annotation Quality')
    plt.ylabel('Improvement Over Vanilla IS (%)')
    plt.title('OPE Improvement by Annotation Quality')
    plt.xticks(x, scenarios)
    plt.legend()
    
    plt.savefig('annotation_quality_results.png')
    plt.close()
    
    #Graph 3: Budget allocation results
    plt.figure(figsize=(10, 6))
    percentages = list(allocation_results.keys())
    isplus_rmses = [allocation_results[p]['ISplus_rmse'] for p in percentages]
    dmplus_rmses = [allocation_results[p]['DMplus_IS_rmse'] for p in percentages]
    
    plt.plot(percentages, isplus_rmses, 'o-', label='IS+')
    plt.plot(percentages, dmplus_rmses, 's-', label='DM+-IS')
    
    plt.xlabel('Percentage of Budget for Expert Annotations')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Expert Budget Allocation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('allocation_results.png')
    plt.close()
    
    print("\nResults Summary:")
    print("-" * 50)
    
    #Find best reward structure, annotation quality, and allocation strategy 
    best_reward = max(reward_results.items(), key=lambda x: x[1]['dmplus_improvement'])[0]
    print(f"Best reward structure: {best_reward}")
    best_annotation = max(annotation_results.items(), key=lambda x: x[1]['dmplus_improvement'])[0]
    print(f"Best annotation quality: {best_annotation}")
    best_allocation_isplus = min(allocation_results.items(), key=lambda x: x[1]['ISplus_rmse'])[0]
    best_allocation_dmplus = min(allocation_results.items(), key=lambda x: x[1]['DMplus_IS_rmse'])[0]
    print(f"Best allocation for IS+: {best_allocation_isplus}%")
    print(f"Best allocation for DM+-IS: {best_allocation_dmplus}%")
    
    return {
        'reward_results': reward_results,
        'annotation_results': annotation_results,
        'allocation_results': allocation_results
    }

if __name__ == "__main__":
    print("Running focused environment parameter exploration...")
    results = explore_key_parameters()
    print("Exploration complete. Check the generated PNG files for visualizations.")