import gc
import itertools
import math
import sys
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import get_arguments, DF, METRICS_MAPPING, mkdir_p, compute_angles
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
try:
    from Models.llm_embeddings import build_embedder, _derive_model_label
    print("Successfully imported build_embedder.")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


OVERLAP_SENTENCES_COLUMNS = ["S0", "S1", "S2"]
INTERSECTION_COLS = ["S0", "S1", "S2"]
FLAG_COL = "duplicate"


# TODO: Yash, you can probably put this part in utils so that it can be reused since there'll be duplicates
def read_final_combined_data():
    pt = Path("./data/final_combined_data.xlsx")
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

    if sys.gettrace() is None:  # Normal mode
        assert len(out) == len(final_combined_data)

    print(f"Length of original DF: {len(df)}, Length of filtered DF: {len(out)} ")
    return out


def load_overlap_data() -> DF:
    """Load the data corresponding to overlap. Paths are hardcoded"""
    data_pt = Path(
        "./data/intersection_analysis.xlsx"
    )

    # Read small number of rows in debug mode
    nrows = None
    if sys.gettrace() is not None:  # Debug
        nrows = 49

    data = pd.read_excel(
        data_pt, sheet_name="Sheet1", usecols=OVERLAP_SENTENCES_COLUMNS, nrows=nrows
    )

    return data


def main():
    args = get_arguments()
    if sys.gettrace() is not None:  # Debug
        model_name = "Qwen/Qwen3-Embedding-8B"
        model = build_embedder(model_name=model_name)
        print("Debug mode: Using Qwen3-Embedding-8B model for faster testing.")
    else: 
        model_name = args.model
        model = build_embedder(
            model_name=model_name,
            device=args.gpu
        )
    model_id = _derive_model_label(model_name, None)
    
    out_dir = Path(args.output_dir) / f"overlap_results/{model_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if sys.gettrace() is not None:  # Debug
        args.model = "qwen3embedding"
        args.batch_size = 2
        out_dir = mkdir_p(Path("../temp/").joinpath("overlap_results"))

    # Data
    data_df = load_overlap_data()
    data_df = filter_duplicates(data_df)
    # if sys.gettrace() is not None:  # Debug
    #     data_df = data_df.head(10)
        
    # n_chunks = len(data_df) // args.batch_size
    # n_chunks = 1 if not n_chunks else n_chunks
    n_chunks = max(1, math.ceil(len(data_df) / args.batch_size))

    # Metrics
    metrics_map = METRICS_MAPPING

    # Iterate over all the models
    model_results = [
        data_df,
    ]

    print(f"\nModel: {model_id}")
    model_time = time.process_time()
    # Iterate over batch of data and generate results
    results = defaultdict(list)
    angle_results = []
    for b_idx, chunk in enumerate(np.array_split(data_df, n_chunks)):
        batch_time = time.process_time()
        # Encode S0, S1 and S2 for data chunk
        chunk_embeds = {
            key: model.encode(chunk[key].to_list(), batch_size=args.batch_size)
            for key in OVERLAP_SENTENCES_COLUMNS
        }

        # Compute all metric results for all the pairs/combinations of 2
        for metric_id, metric_cls in metrics_map.items():
            for pair_embeds in itertools.combinations(chunk_embeds.keys(), 2):
                pair_id = "_".join(pair_embeds)
                metric_inputs = [chunk_embeds[p_embed] for p_embed in pair_embeds]
                results[f"{model_id}_{metric_id}_{pair_id}"].append(
                    metric_cls(*metric_inputs)
                )

        # Generate the projection results
        angle_results.append(
            compute_angles(
                chunk_embeds["S0"], chunk_embeds["S1"], chunk_embeds["S2"]
            )
        )

    print(
        f"Model: {model_id} Time: {time.process_time() - batch_time}"
    )

    # H1 Results per model
    out = {}
    for key, val in results.items():
        out[key] = np.concatenate(val).squeeze()
    model_results.append(pd.DataFrame.from_dict(out))

    # Delete the model and free the gpu memory. Required for LLMs
    if hasattr(model, 'model'):
        model.model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Print Time taken for each model
    print(
        f"Model: {model_id}, Time Taken: {time.process_time() - model_time:.2f} s"
    )

    # H1 Results Across model
    out = pd.concat(model_results, axis=1)
    # ensure directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_excel(out_dir / "h1_results.xlsx", index=False)

    print("Done")

if __name__ == "__main__":
    program_time = time.process_time()
    main()
    print(f"Done! Time Taken: {time.process_time() - program_time:.2f} s")
