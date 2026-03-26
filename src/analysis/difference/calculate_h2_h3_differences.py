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
    output_dir = f'Results/difference_results/{model_id}/Table15'
    os.makedirs(output_dir, exist_ok=True)

    # Read the data
    df = pd.read_excel(input_file)

    metric = 'cos'
    col_bd = f'{model_id}_{metric}_B_D'
    col_ad = f'{model_id}_{metric}_A_D'
    col_ab = f'{model_id}_{metric}_A_B'
    print(df.columns)
    if col_bd in df.columns and col_ad in df.columns and col_ab in df.columns:
        bd_mean = df[col_bd].mean()
        diff1_mean = (df[col_ad] - df[col_bd]).mean()
        diff2_mean = (df[col_ab] - df[col_bd]).mean()
        results.append({
            'Model': model_id,
            f'{metric}(B,D)': f"{bd_mean:.4f}",
            f'{metric}(A,D)-{metric}(B,D)': f"{diff1_mean:.4f}",
            f'{metric}(A,B)-{metric}(B,D)': f"{diff2_mean:.4f}",
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