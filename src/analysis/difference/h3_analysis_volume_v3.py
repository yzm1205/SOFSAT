from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from utils import mkdir_p, DF, MODEL_LIST, METRIC_LIST
from constants import remove_common_rows_from_df
from analysis.overlap.C1_analysis_volume_v3 import EXP_KEY_MAPPING, _METRICS

DIFFERENCE_COLS = ["A", "B", "D"]


def get_pairwise_analysis(
    data: DF,
    metric: str,
    metric_type: str,
    eps1: float,
    eps2: float,
) -> Dict[str, float]:
    diff1, diff2 = _get_diff1_diff2(data, metric_name=metric, metric_type=metric_type)

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
    # save_pt: Path,
    metric: str,
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
                        sb_data, metric, metric_type, eps1, eps2
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


def filter_duplicates(df: DF, key: str) -> DF:
    assert key in ["left", "right"]
    df = df.drop_duplicates(subset=DIFFERENCE_COLS)

    if key == "left":
        removed_samples = pd.read_excel(
            "./data/final_removed_left_difference_data.xlsx"
        )
    elif key == "right":
        removed_samples = pd.read_excel(
            "./data/final_removed_right_difference_data.xlsx"
        )
    else:
        raise ValueError(f"Not Implemented")

    # Remove unwanted samples
    out = remove_common_rows_from_df(df, removed_samples)

    print(f"Original Length: {len(df)}, Filter Length: {len(out)}")
    return out


def main(
    exp_key: str,
    metric_bounds,
    grid_points: int = 11,
    n_bins: int = 11,
    save: bool = True,
):
    exp_key_mapping = EXP_KEY_MAPPING

    models = MODEL_LIST
    models = models[exp_key_mapping[exp_key].model_key]

    # Metric List
    metrics = _METRICS

    data_pt = Path(f"./Results/difference_results/h2_and_h3_results.xlsx")


    save_dir = mkdir_p(data_pt.parent.joinpath("Difference_Analysis_Hypothesis_3"))
    if sys.gettrace() is not None:  # Debug
        save_dir = mkdir_p(
            Path(
                "./temp/difference_results/Difference_Analysis_Hypothesis_3"
            )
        )

    print(f"Saving to directory: {str(save_dir)}, Grid Points: {grid_points}")

    def analyze(diff_data: DF, data_key: str) -> None:
        model_metric_results = {}
        for model_name, model_key in models.items():
            metric_results = {}
            for metric, metric_type in metrics.items():
                print(f"Model: {model_name}, Metric: {metric}")
                sb_data = diff_data[
                    [
                        col
                        for col in diff_data.columns
                        if col.startswith(f"{model_key}_{metric}")
                    ]
                ]
                res = model_analysis(
                    sb_data,
                    metric=metric,
                    metric_type=metric_type,
                    metric_bound=metric_bounds[metric],
                    grid_points=grid_points,
                    n_bins=n_bins,
                )

                print(
                    f"Data Key: {data_key}, Metric: {metric}, Model: {model_name}, Results: {res}"
                )
                metric_results[metric] = res
            model_metric_results[model_name] = metric_results

        print("Saving")
        if save:
            if sys.gettrace() is None:  # Normal
                # Assuming no common keys between sb_means and sb_counts
                with open(
                    save_dir.joinpath(
                        f"1Volume_{data_key}_difference_analysis_results.json"
                    ),
                    "w",
                ) as outfile:
                    json.dump(model_metric_results, outfile)

    nrows = None
    if sys.gettrace() is not None:
        nrows = 100

    # Analyze Left Difference Data
    sheet = "LDiff"
    ldiff_data = pd.read_excel(data_pt, sheet_name=sheet, nrows=nrows)
    if sys.gettrace() is None:  # Normal
        print("Filtering duplicates")
        ldiff_data = filter_duplicates(ldiff_data, key="left")
    # analyze(ldiff_data, sheet)

    # Analyze Right Difference Data
    sheet = "RDiff"
    rdiff_data = pd.read_excel(data_pt, sheet_name=sheet, nrows=nrows)
    if sys.gettrace() is None:  # Normal
        print("Filtering duplicates")
        rdiff_data = filter_duplicates(rdiff_data, key="right")
    # analyze(rdiff_data, sheet)

    # Analyze Combined Difference Data
    combined_data = pd.concat([ldiff_data, rdiff_data])
    print(f"Combined Data: {len(combined_data)}")
    analyze(combined_data, "full")

    print("Done")


def _get_diff1_diff2(
    data: DF, metric_name: str, metric_type: str
) -> Tuple[pd.Series, pd.Series]:
    a_d = [col for col in data.columns if col.lower().endswith(f"{metric_name}_a_d")][0]
    b_d = [col for col in data.columns if col.lower().endswith(f"{metric_name}_b_d")][0]
    a_b = [col for col in data.columns if col.lower().endswith(f"{metric_name}_a_b")][0]

    diff1 = data[a_d] - data[b_d]
    diff2 = data[a_b] - data[b_d]

    if metric_type == "sim":
        return diff1, diff2
    elif metric_type == "distance":
        return -1 * diff1, -1 * diff2
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


def _get_lower_upper_bounds_across_models(metrics):
    _OLD_DP = Path(
        "/home/yash/set_theoretic/test_auto_encoder/Results/SanjeevPaper/difference_analysis.xlsx"
    )
    _BDI_DP = Path(
        "/home/yash/set_theoretic/sanjeev_extra_experiments/Results/difference_results/h2_and_h3_results.xlsx"
    )  # BDI
    _NLP_DP = Path(
        "/home/yash/set_theoretic/sanjeev_extra_experiments/Results_NLP/difference_results/h2_and_h3_results.xlsx"
    )  # NLP
    _2026_DP = Path(
        "/home/yash/set_theoretic/Results/difference_results/h2_and_h3_results.xlsx"
    )  # 2026
    _ALL_MODELS = {
        # "Qwen3Emb": ("qwen3embedding", _2026_DP),
        # "Mistrale5": ("mistrale5", _2026_DP),
        # "Octenaembedding": ("octenaembedding", _2026_DP),
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
    }

    if sys.gettrace() is not None:
        _ALL_MODELS = {
            "SBert": ("sb", _OLD_DP),
            "RoBERTa": ("roberta", _BDI_DP),
            "LLaMA3": ("llama3", _NLP_DP),
        }

    def _read_data(data_pt: Path):
        # Analyze Left Difference Data
        nrows = None
        if sys.gettrace() is not None:
            nrows = 100

        sheet = "LDiff"
        ldiff_data = pd.read_excel(data_pt, sheet_name=sheet, nrows=nrows)
        if sys.gettrace() is None:  # Normal
            print("Filtering duplicates")
            ldiff_data = filter_duplicates(ldiff_data, key="left")
        # analyze(ldiff_data, sheet)

        # Analyze Right Difference Data
        sheet = "RDiff"
        rdiff_data = pd.read_excel(data_pt, sheet_name=sheet, nrows=nrows)
        if sys.gettrace() is None:  # Normal
            print("Filtering duplicates")
            rdiff_data = filter_duplicates(rdiff_data, key="right")
        # analyze(rdiff_data, sheet)

        # Analyze Combined Difference Data
        combined_data = pd.concat([ldiff_data, rdiff_data])
        print(f"Combined Data: {len(combined_data)}")

        return combined_data

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
                model_data = _read_data(m_dp)

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
            diff1, diff2 = _get_diff1_diff2(model_data, metric_name, metric_type)

            eps1_lb, eps1_ub = diff1.min(), diff1.max()
            eps2_lb, eps2_ub = diff2.min(), diff2.max()

            eps1_lb_list.append(eps1_lb)
            eps1_ub_list.append(eps1_ub)
            eps2_lb_list.append(eps2_lb)
            eps2_ub_list.append(eps2_ub)

        eps1_lb, eps1_ub = min(eps1_lb_list), max(eps1_ub_list)
        eps2_lb, eps2_ub = min(eps2_lb_list), max(eps2_ub_list)

        metric_bounds[metric_name] = ((eps1_lb, eps1_ub), (eps2_lb, eps2_ub))

    return metric_bounds


def run():
    n_bins = 11
    grid_points = 11

    # Select Metrics
    metrics = _METRICS

    metric_bounds = _get_lower_upper_bounds_across_models(metrics)
    print(f"Metric Bounds: {metric_bounds}")
    for kk_idx, kk_key in enumerate(EXP_KEY_MAPPING.keys()):
        main(kk_key, metric_bounds, grid_points, save=True, n_bins=n_bins)


if __name__ == "__main__":
    start_time = time.perf_counter()
    run()
    print(f"Time Taken: {time.perf_counter() - start_time}")
