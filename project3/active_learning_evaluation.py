from ml_backbone import ml_manager 


# --- INITIAL BENCHMARKING ---
# Call disagreement metrics right away to see the baseline gap
initial_disagreement = ml_manager.get_disagreement_metrics()
print(f"Initial Baseline Accuracy: {ml_manager.baseline_acc:.4f}")
print(f"Initial Deferral Disagreement Rate: {initial_disagreement['disagreement_rate']:.4f}\n")

# Active Learning Simulation Loop 
print("Starting Active Learning Queries...")
num_queries_to_run = 30

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
    
    # Log the progress every few steps
    print(f"Query {i+1}/{num_queries_to_run} | "
          f"Acc: {update_result['current_accuracy']:.4f} | "
          f"Accuracy Gain Per Query: {update_result['accuracy_gain_per_query']:.4f} | "
          f"Remaining Deferral Ops: {update_result['total_deferral_opportunities']}")

# Final Visualization of the Active Learning Process 
print("\nPlotting results...")
ml_manager.plot_metrics()