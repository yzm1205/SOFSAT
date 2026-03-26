import pandas as pd
import os

# Read the Excel file
df = pd.read_excel('Results/overlap_results/h1_results.xlsx')

# Define the models and metrics
models = ['qwen3embedding', 'mistrale5', 'octenaembedding']
metrics = ['cos', 'normalized_l1', 'normalized_l2', 'l1', 'l2', 'nsed', 'dot']

# Directory to save results
output_dir = 'Results/overlap_results/Intersection_Analysis_Hypothesis_1/Table1'
os.makedirs(output_dir, exist_ok=True)

# For each metric, collect data
for metric in metrics:
    results = []
    for model in models:
        col_s0_s1 = f'{model}_{metric}_S0_S1'
        col_s0_s2 = f'{model}_{metric}_S0_S2'
        col_s1_s2 = f'{model}_{metric}_S1_S2'
        
        # Check if columns exist
        if col_s0_s1 in df.columns and col_s0_s2 in df.columns and col_s1_s2 in df.columns:
            # Calculate differences
            diff1 = df[col_s0_s1] - df[col_s1_s2]
            diff2 = df[col_s0_s2] - df[col_s1_s2]
            
            # Calculate std
            std_diff1 = diff1.std()
            std_diff2 = diff2.std()
            
            results.append({'model': model, 'A0_minus_AB': f"{diff1.mean():.4f} ± {std_diff1:.4f}", 'BO_minus_AB': f"{diff2.mean():.4f} ± {std_diff2:.4f}"})
        else:
            print(f'Columns for {model}_{metric} not found.')
    
    # Create DataFrame and save to CSV
    if results:
        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_dir, f'{metric}_mean_std.csv')
        results_df.to_csv(output_file, sep='&', index=False)
        print(f'Saved {output_file}')