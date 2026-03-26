import pandas as pd
import os

# Input data file and output directory
input_file = 'Results/difference_results/h2_and_h3_results.xlsx'
output_dir = 'Results/difference_results/Difference_Analysis_Hypothesis_3/Table15'
os.makedirs(output_dir, exist_ok=True)

# Read the data
df = pd.read_excel(input_file)

models = ['qwen3embedding', 'mistrale5', 'octenaembedding']
metrics = ['normalized_l1', 'normalized_l2', 'l1', 'l2', 'nsed', 'cos', 'dot']

for metric in metrics:
    results = []
    for model in models:
        col_bd = f'{model}_{metric}_B_D'
        col_ad = f'{model}_{metric}_A_D'
        col_ab = f'{model}_{metric}_A_B'
        if col_bd in df.columns and col_ad in df.columns and col_ab in df.columns:
            bd_mean = df[col_bd].mean()
            diff1_mean = (df[col_ad] - df[col_bd]).mean()
            diff2_mean = (df[col_ab] - df[col_bd]).mean()
            results.append({
                'Model': model,
                f'{metric}(B,D)': f"{bd_mean:.4f}",
                f'{metric}(A,D)-{metric}(B,D)': f"{diff1_mean:.4f}",
                f'{metric}(A,B)-{metric}(B,D)': f"{diff2_mean:.4f}",
            })
        else:
            print(f'Missing columns for {model} {metric}')
    
    # Save results for this metric
    if results:
        results_df = pd.DataFrame(results)
        out_path = os.path.join(output_dir, f'{metric}_diffs.csv')
        results_df.to_csv(out_path, sep='&', index=False)
        print(f'Saved {out_path}')
    else:
        print(f'No results for {metric}')