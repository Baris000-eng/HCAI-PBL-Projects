from ml_backbone import ml_manager 


initial_disagreement = ml_manager.get_disagreement_metrics()
print(f"Baseline Model Accuracy: {ml_manager.baseline_acc:.4f}")
print(f"Simulated Expert Accuracy: {ml_manager.expert_acc:.4f}")
print(f"Initial Deferral Agreement Rate: {initial_disagreement['agreement_rate']:.4f}")
print(f"Initial Deferral Disagreement Rate: {initial_disagreement['disagreement_rate']:.4f}\n")


# Active Learning Simulation Loop 
print("Starting Active Learning Queries...")
num_queries_to_run = 30
print(f"Number of active learning queries to run: {num_queries_to_run}")

initial_al_accuracy = ml_manager.accuracy_history[0]
print(f"The active learning model's accuracy at Query 0 (Initialization) is: {initial_al_accuracy:.4f}\n")

for i in range(num_queries_to_run):
    # Get the next most uncertain sample
    sample_data = ml_manager.get_next_uncertain_sample()
    if sample_data is None:
        break
        
    # Pass the expert's label into the update system 
    update_result = ml_manager.process_query_update(
        idx=sample_data['sample_index'], 
        chosen_label=sample_data['simulated_expert_label']
    )
    
    # Log the progress 
    print(f"Query {i+1}/{num_queries_to_run} | "
          f"Acc: {update_result['current_test_accuracy']:.4f} | "
          f"Accuracy Gain/Loss Per Query: {update_result['accuracy_gain_per_query']:.4f} | "
          f"Remaining Deferral Ops: {update_result['total_deferral_opportunities']}")

# Final Visualization of the Active Learning Process 
print("\nPlotting results...")
ml_manager.plot_metrics()