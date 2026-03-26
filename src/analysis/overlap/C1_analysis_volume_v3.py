from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Dict, Tuple
import sys

import numpy as np
import pandas as pd

from helper import FLAG_COL, INTERSECTION_COLS

from utils import DF, mkdir_p, MODEL_LIST

_METRICS = {
    # "nsed": "distance",
    "cos": "sim",
}


@dataclass
class ExpKeys:
    data_path_key: str
    model_key: str


EXP_KEY_MAPPING: Dict[str, ExpKeys] = {
    # "bdi": ExpKeys(data_path_key="", model_key="classical"),
    # "nlp": ExpKeys(data_path_key="_NLP", model_key="llm"),
    # "old": ExpKeys(data_path_key="_old", model_key="old"),
    "2026": ExpKeys(data_path_key="", model_key="2026"),
}


def read_data(filename: Path, sheet: str) -> DF:
    nrows = None
    # if sys.gettrace() is not None:  # Debug
    #     nrows = 100

    data = pd.read_excel(filename, sheet_name=sheet, nrows=nrows)
    return data


def get_pairwise_analysis(
    data: DF,
    metric_type: str,
    eps1: float,
    eps2: float,
) -> Dict[str, float]:
    assert len(data.columns) == 3, "Dataframe must have 3 columns"

    diff1, diff2 = _get_diff1_diff2(data, metric_type)

    # Find Rows where both conditions hold i.e. S0_S1 > S1_S2 and S0_S2 > S1_S2
    true_rows = ((diff1 >= eps1) & (diff2 >= eps2)).sum()

    # Find Rows where one condition holds i.e. S0_S1 > S1_S2
    s1_true = ((diff1 >= eps1) & (diff2 < eps2)).sum()

    # Find Rows where one condition holds i.e. S0_S2 > S1_S2
    s2_true = ((diff1 < eps1) & (diff2 >= eps2)).sum()

    # Find Rows where none of the conditions hold
    false_rows = ((diff1 < eps1) & (diff2 < eps2)).sum()

    assert (true_rows + s1_true + s2_true + false_rows) - len(
        data
    ) == 0, "Should sum to 0"

    return {
        "true_rows": true_rows / len(data),
        "s1_true": s1_true / len(data),
        "s2_true": s2_true / len(data),
        "false_rows": false_rows / len(data),
    }


def model_analysis(
    sb_data: DF,
    metric_type: str,
    metric_bound: Tuple[Tuple[float, float], Tuple[float, float]],
    grid_points: int = 101,
    n_bins: int = 11,
) -> Dict[str, float]:

    (min_eps1_lb, max_eps1_ub), (min_eps2_lb, max_eps2_ub) = metric_bound

    # Take odd samples to get even number of bins
    n_bins = n_bins if n_bins % 2 != 0 else n_bins + 1

    # Make grod points also odd
    grid_points = grid_points if grid_points % 2 != 0 else grid_points + 1

    # Make Bins like Yash made
    x_bin_ticks = np.linspace(min_eps1_lb, max_eps1_ub, n_bins)
    y_bin_ticks = np.linspace(min_eps2_lb, max_eps2_ub, n_bins)

    x_ranges = [(i, j) for i, j in zip(x_bin_ticks[:-1], x_bin_ticks[1:])]
    y_ranges = [(i, j) for i, j in zip(y_bin_ticks[:-1], y_bin_ticks[1:])]

    per_bin_values = []
    for eps1_lb, eps1_ub in x_ranges:
        for eps2_lb, eps2_ub in y_ranges:

            # Now Compute the volume under the surface (or average z's) for this rectanglular bin
            eps1_values = np.linspace(eps1_lb, eps1_ub, grid_points)
            eps2_values = np.linspace(eps2_lb, eps2_ub, grid_points)

            out = []
            for eps1 in eps1_values:
                for eps2 in eps2_values:
                    eps1_eps2_out = get_pairwise_analysis(
                        sb_data, metric_type, eps1, eps2
                    )
                    out.append(eps1_eps2_out)

            ## Avg Z values
            avg_true_samples = {}
            for key in out[0].keys():
                vals = [ii[key] for ii in out]
                avg_true_samples[key] = np.mean(vals)

            assert abs(sum([v for v in avg_true_samples.values()]) - 1) < 1e-6
            per_bin_values.append(avg_true_samples)

    final_res = {}
    for key in per_bin_values[0].keys():
        vals = [ii[key] for ii in per_bin_values]
        final_res[key] = np.mean(vals).item()

    assert abs(sum([v for v in final_res.values()]) - 1) < 1e-6
    return final_res


def read_final_combined_data():
    pt = Path("./data/final_combined_data.xlsx") #input sample
    return pd.read_excel(pt)


def filter_duplicates(df: DF) -> DF:
    final_combined_data = read_final_combined_data()

    # Keep the one where either of the consecutive sentences pairs from (prev, curr, next) are not same.
    final_combined_data = final_combined_data[final_combined_data[FLAG_COL] == 0]
    final_combined_data = final_combined_data[INTERSECTION_COLS]

    # Now, filter the results dataframe accordingly
    out = df.merge(
        final_combined_data, on=INTERSECTION_COLS, how="inner", indicator=True
    )
    out = out.drop_duplicates(subset=INTERSECTION_COLS)
    out = out[out["_merge"] == "both"]
    out = out.drop("_merge", axis=1)

    if sys.gettrace() is None:  # Normal
        assert len(out) == len(final_combined_data)

    print(f"Length of original DF: {len(df)}, Length of filtered DF: {len(out)} ")
    return out


def _get_diff1_diff2(data: DF, metric_type: str) -> Tuple[pd.Series, pd.Series]:
    s0_s1 = [col for col in data.columns if "s0_s1" in col.lower()][0]
    s0_s2 = [col for col in data.columns if "s0_s2" in col.lower()][0]
    s1_s2 = [col for col in data.columns if "s1_s2" in col.lower()][0]

    diff1 = data[s0_s1] - data[s1_s2]
    diff2 = data[s0_s2] - data[s1_s2]

    if metric_type == "sim":
        return diff1, diff2
    elif metric_type == "distance":
        return -1 * diff1, -1 * diff2
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


def _get_lower_upper_bounds_across_models(metrics):
    _2026_DP = Path(f"./Results/overlap_results/h1_results.xlsx")
    _ALL_MODELS = {
        "SBert": ("sb", _OLD_DP),
        "Laser": ("lsr", _OLD_DP),
        "USE": ("use", _OLD_DP),
        "GPT": ("gpt", _OLD_DP),
        "Llama2": ("lma", _OLD_DP),
        "RoBERTa": ("roberta", _BDI_DP),
        "MPNet": ("mpnet", _BDI_DP),
        "SimCSE": ("simcse", _BDI_DP),
        "InferSent": ("infersent", _BDI_DP),
        "Mistral": ("mistral", _NLP_DP),
        "LLaMA3": ("llama3", _NLP_DP),
        "Olmo": ("olmo", _NLP_DP),
        "OpenELM": ("openelm", _NLP_DP),
        # "Qwen3Emb": ("qwen3embedding", _2026_DP),
        # "Mistrale5": ("mistrale5", _2026_DP),
        # "Octenaembedding": ("octenaembedding", _2026_DP)
    }

    if sys.gettrace() is not None:
        _ALL_MODELS = {
            "Qwen3Emb": ("qwen3embedding", _2026_DP),
            "Mistrale5": ("mistrale5", _2026_DP),
            "Octenaembedding": ("octenaembedding", _2026_DP)
            # "SBert": ("sb", _OLD_DP),
            # "RoBERTa": ("roberta", _BDI_DP),
            # "LLaMA3": ("llama3", _NLP_DP),
        }

    # Get lower and upper bounds for a metric across all models
    seen_data_pt = {}
    all_models = _ALL_MODELS
    metric_bounds = {}
    for metric_name, metric_type in metrics.items():
        eps1_lb_list = []
        eps1_ub_list = []
        eps2_lb_list = []
        eps2_ub_list = []
        for m_name, m_val in all_models.items():
            m_key, m_dp = m_val

            # Don't read data again and again
            if m_dp not in seen_data_pt.keys():
                model_data = read_data(m_dp, sheet="Sheet1")
                # if sys.gettrace() is None:  # Normal Mode
                print("Filtering duplicates")
                model_data = filter_duplicates(model_data)

                seen_data_pt[m_dp] = model_data
            else:
                model_data = seen_data_pt[m_dp]

            model_data = model_data[
                [
                    col
                    for col in model_data.columns
                    if col.startswith(f"{m_key}_{metric_name}")
                ]
            ]
            print(f"Model: {m_name}, Metric: {metric_name}, Data Shape: {model_data.shape}")
            diff1, diff2 = _get_diff1_diff2(model_data, metric_type)

            eps1_lb, eps1_ub = diff1.min(), diff1.max()
            eps2_lb, eps2_ub = diff2.min(), diff2.max()

            eps1_lb_list.append(eps1_lb)
            eps1_ub_list.append(eps1_ub)
            eps2_lb_list.append(eps2_lb)
            eps2_ub_list.append(eps2_ub)

        # Find the range that covers all the points from all the models
        eps1_lb, eps1_ub = min(eps1_lb_list), max(eps1_ub_list)
        eps2_lb, eps2_ub = min(eps2_lb_list), max(eps2_ub_list)

        metric_bounds[metric_name] = ((eps1_lb, eps1_ub), (eps2_lb, eps2_ub))

    return metric_bounds


def main(
    exp_key: str,
    metric_bounds,
    grid_points: int = 11,
    n_bins: int = 11,
    save: bool = True,
):
    # exp_key = "old"
    exp_key_mapping = EXP_KEY_MAPPING

        
    data_pt = Path(f"./Results/overlap_results/h1_results.xlsx")

    save_dir = mkdir_p(data_pt.parent.joinpath("Intersection_Analysis_Hypothesis_1"))
    if sys.gettrace() is not None:  # Debug
        save_dir = Path(
            "./temp/overlap_results/Intersection_Analysis_Hypothesis_1"
        )

    print(f"Saving to directory: {str(save_dir)}, Grid Points: {grid_points}")

    # Read Data
    data = read_data(data_pt, sheet="Sheet1")
    if sys.gettrace() is None:  # Normal Mode
        print("Filtering duplicates")
        data = filter_duplicates(data)

    # Select Models
    # models = models["classical"]  # BDI
    # models = models["llm"]  # NLP
    models = MODEL_LIST
    models = models[exp_key_mapping[exp_key].model_key]

    # Get Metrics
    metrics = _METRICS

    model_metric_results = {}
    for model_name, model_key in models.items():
        metric_results = {}
        for metric, metric_type in metrics.items():
            print(f"Model: {model_name}, Metric: {metric}")
            sb_data = data[
                [col for col in data.columns if col.startswith(f"{model_key}_{metric}")]
            ]
            res = model_analysis(
                sb_data,
                # mkdir_p(
                #     save_dir.joinpath(
                #         f"1Volume_results_model_{model_name}_metric_{metric}.xlsx"
                #     )
                # ),
                metric_type=metric_type,
                metric_bound=metric_bounds[metric],
                grid_points=grid_points if metric == "nsed" else 2 * grid_points,
                n_bins=n_bins,
            )

            print(f"Metric: {metric}, Model: {model_name}, Results: {res}")
            metric_results[metric] = res
        model_metric_results[model_name] = metric_results

    if save:
        if sys.gettrace() is None:  # Normal
            # Assuming no common keys between sb_means and sb_counts
            print("Saving File")
            with open(
                save_dir.joinpath(f"1Volume_intersection_analysis_results.json"), "w"
            ) as outfile:
                json.dump(model_metric_results, outfile)

    print("Done")


def run():
    # Select Metrics
    n_bins = 11
    grid_points = 11

    print(f"n_bins: {n_bins}, grid_points: {grid_points}")

    metrics = _METRICS
    metric_bounds = _get_lower_upper_bounds_across_models(metrics)
    print(f"Metric Bounds: {metric_bounds}")

    for idx, kk_key in enumerate(EXP_KEY_MAPPING.keys()):
        main(kk_key, metric_bounds, grid_points, save=True, n_bins=n_bins)


if __name__ == "__main__":
    start_time = time.perf_counter()
    run()
    print(f"Time Taken: {time.perf_counter() - start_time}")
