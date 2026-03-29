import pandas as pd
import os
import sys
sys.path.insert(0,"./src/")
from utils import get_arguments, derive_model_label

def main():
    args = get_arguments() 
    model_id = derive_model_label(args.model, None)
    # Read the Excel file
    df = pd.read_excel(f'Results/overlap_results/{model_id}/h1_results.xlsx')
    metric = 'cos'

    # Directory to save results
    output_dir = f'Results/overlap_results/{model_id}/Table1'
    os.makedirs(output_dir, exist_ok=True)

    # For each metric, collect data
    
    results = []
    
    col_s0_s1 = f'{model_id}_{metric}_S0_S1'
    col_s0_s2 = f'{model_id}_{metric}_S0_S2'
    col_s1_s2 = f'{model_id}_{metric}_S1_S2'
    
    # Check if columns exist
    if col_s0_s1 in df.columns and col_s0_s2 in df.columns and col_s1_s2 in df.columns:
        # Calculate differences
        diff1 = df[col_s0_s1] - df[col_s1_s2]
        diff2 = df[col_s0_s2] - df[col_s1_s2]
        
        # Calculate std
        std_diff1 = diff1.std()
        std_diff2 = diff2.std()
        
        results.append({'model': model_id, 'A0_minus_AB': f"{diff1.mean():.4f} ± {std_diff1:.4f}", 'BO_minus_AB': f"{diff2.mean():.4f} ± {std_diff2:.4f}"})
    else:
        print(f'Columns for {model_id}_{metric} not found.')
    
    # Create DataFrame and save to CSV
    if results:
        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_dir, f'{metric}_mean_std.csv')
        results_df.to_csv(output_file, sep='&', index=False)
        print(f'Saved {output_file}')
            
if __name__ == "__main__":
    args = get_arguments()
    main()