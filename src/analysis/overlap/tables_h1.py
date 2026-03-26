from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import List, Dict

import pandas as pd
sys.path.insert(0,"/home/yash/set_theoretic/src")
from utils import MODEL_LIST, METRIC_LIST, mkdir_p


def read_json(data_pt: Path):
    with open(data_pt, "r") as f:
        return json.load(f)


def table2(
    data_pt: Path,
    out_pt: Path,
    models: Dict[str, str],
    metrics: List[str],
    round_digits: int = 4,
):
    data = read_json(data_pt)
    for metric in metrics:
        results = defaultdict(dict)
        for model_key, model_val in models.items():
            s0_s1 = data[model_key][metric][f"{model_val}_{metric}_S0_S1"]
            s0_s2 = data[model_key][metric][f"{model_val}_{metric}_S0_S2"]
            s1_s2 = data[model_key][metric][f"{model_val}_{metric}_S1_S2"]

            results[f"{model_key}"]["A_B"] = round(s1_s2, round_digits)
            results[f"{model_key}"]["A_O_minus_A_B"] = round(
                (s0_s1 - s1_s2), round_digits
            )
            results[f"{model_key}"]["B_O_minus_A_B"] = round(
                (s0_s2 - s1_s2), round_digits
            )

        # Save Results
        results = pd.DataFrame.from_dict(results, orient="index")

        f_name = mkdir_p(out_pt.joinpath(f"{metric}.csv"))
        results.to_csv(f"{f_name}", sep="&")

    print("Done")


def table3(data_pt: Path, out_pt: Path, models: Dict[str, str], metrics: List[str]):
    data = read_json(data_pt)
    for metric in metrics:
        results = defaultdict(dict)
        for model_key, model_val in models.items():
            true_rows = round(data[model_key][metric]["true_rows"][-1], 2)
            s1_true = round(data[model_key][metric]["s1_true"][-1], 2)
            s2_true = round(data[model_key][metric]["s2_true"][-1], 2)
            false_rows = round(data[model_key][metric]["false_rows"][-1], 2)

            results[f"{model_key}"]["true_rows"] = true_rows
            results[f"{model_key}"]["s1_true"] = s1_true
            results[f"{model_key}"]["s2_true"] = s2_true
            results[f"{model_key}"]["false_rows"] = false_rows

        # Save Results
        results = pd.DataFrame.from_dict(results, orient="index")
        f_name = mkdir_p(out_pt.joinpath(f"{metric}.csv"))
        results.to_csv(f"{f_name}", sep="&")

    print("Done")

def main():
    
    data_pt = Path(
        f"./Results/overlap_results/"
        f"Intersection_Analysis_Hypothesis_1/intersection_analysis_results.json"
    )  # Local

    out_pt = data_pt.parent

    if sys.gettrace() is not None:
        out_pt = mkdir_p(
            "./Results/temp/Intersection_Analysis_Hypothesis_1"
        )

    models = MODEL_LIST["2026"]  # 2026
    
    metrics = list(METRIC_LIST.keys())

    # Get Results in required format
    table2(data_pt, mkdir_p(out_pt.joinpath("Table13")), models, metrics)
    
    
   
    

    # table3(data_pt, mkdir_p(out_pt.joinpath("Table3")), models, metrics)


if __name__ == "__main__":
    program_time = time.process_time()
    main()
    print(f"Done. Total Time: {time.process_time() - program_time}")
    
    # Table 13 in appendix
