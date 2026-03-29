import pandas as pd
import os
import sys
sys.path.insert(0,"./src/")
from utils import get_arguments, derive_model_label


def main():
    args = get_arguments() 
    model_id = derive_model_label(args.model, None)
    results = []
    # Input data file and output directory
    input_file = f'Results/difference_results/{model_id}/C2_and_C3_results.xlsx'
    output_dir = f'Results/difference_results/{model_id}/Table3_alternative'
    os.makedirs(output_dir, exist_ok=True)

    # Read the data
    df = pd.read_excel(input_file)

    metric = 'cos'
    col_abd = f'{model_id}_{metric}_(A-B)_D'
    col_abb = f'{model_id}_{metric}_(A-B)_B'
    print(df.columns)
    if col_abd in df.columns and col_abb in df.columns:
        diff1_mean = (df[col_abd] - df[col_abb]).mean()
        results.append({
            'Model': model_id,
            f'{metric}(E(AB,D))-{metric}(E(AB,B))': f"{diff1_mean:.4f}"
        })
    else:
        print(f'Missing columns for {model_id} {metric}')

    # Save results for this metric
    if results:
        results_df = pd.DataFrame(results)
        out_path = os.path.join(output_dir, f'{metric}_diffs.csv')
        results_df.to_csv(out_path, sep='&', index=False)
        print(f'Saved {out_path}')
    else:
        print(f'No results for {metric}')
        
if __name__ == "__main__":
    args = get_arguments()
    main()