import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import get_arguments, DF, mkdir_p, compute_angles
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
try:
    from Models.llm_embeddings import build_embedder, _derive_model_label
    print("Successfully imported build_embedder.")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
    print("current_device =", torch.cuda.current_device())

UNION_SENTENCES_COLUMNS = ["S1", "S2", "Sy"]

ANGLE_COLUMNS = [
    "Projection Location",
    "Angle_A_B",
    "Angle_A_Projection",
    "Angle_B_Projection",
    "Norm_A",
    "Norm_B",
    "Norm_Proj",
    "Norm_Orig_Vec",
    "TS_Proj_A",
    "TS_Proj_B",
    "TS_A_B",
]


def load_union_data() -> DF:
    """Load the data corresponding to union. Paths are hardcoded"""

    data_pt = Path(
        ".data/union_data.xlsx"
    )

    # Read small number of rows in debug mode
    nrows = None
    if sys.gettrace() is not None:  # Debug
        nrows = 128

    data = pd.read_excel(
        data_pt,
        sheet_name="Union",
        usecols=UNION_SENTENCES_COLUMNS,
        nrows=nrows,
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

    exp_id = "union"
    out_dir = Path(args.output_dir) / f"{exp_id}_results/{model_id}"

    if sys.gettrace() is not None:  # Debug
        args.batch_size = 2
        out_dir = Path("../temp/").joinpath(f"{exp_id}_results")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    data_df = load_union_data()

    chunk_size = getattr(args, "chunk_size", 8)
    encode_batch_size = args.batch_size
    n_chunks = max(1, math.ceil(len(data_df) / chunk_size))

    print(f"\nModel: {model_id}")
    model_time = time.process_time()

    # Iterate over batch of data and generate results
    angle_results = []
    for b_idx, start in enumerate(tqdm(range(0, len(data_df), chunk_size), total=n_chunks)):
        chunk = data_df.iloc[start:start + chunk_size]
        n = len(chunk)

        s1 = chunk["S1"].tolist()
        s2 = chunk["S2"].tolist()
        sy = chunk["Sy"].tolist()

        # Encode S1, S2 and Sy for data chunk in a single call
        with torch.inference_mode():
            all_embeds = model.encode(s1 + s2 + sy, batch_size=encode_batch_size)

        s1_emb = all_embeds[:n]
        s2_emb = all_embeds[n:2*n]
        sy_emb = all_embeds[2*n:3*n]

        # Sanity-check embeddings: must be 2D with one row per sentence
        embed_shape = tuple(all_embeds.shape)
        if len(embed_shape) != 2 or embed_shape[0] != 3 * n:
            raise ValueError(
                f"Unexpected embedding shape from model '{model_id}' (batch={b_idx}): "
                f"{embed_shape}; expected ({3 * n}, D)"
            )

        # Generate the projection results
        angle_results.append(compute_angles(sy_emb, s1_emb, s2_emb))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # H6/Angle Results per model — normalize/validate shapes from compute_angles
    # each entry in angle_results should be a (B,11) array; make robust to (11,), (B,), or (B,1) returns
    if len(angle_results) == 0:
        angle_arr = np.empty((0, 11))
    else:
        proc = []
        for a in angle_results:
            arr = np.asarray(a)
            if arr.ndim == 1:
                # single-row result (11,) -> (1,11); otherwise treat as (N,1)
                if arr.size == 11:
                    arr = arr.reshape(1, 11)
                else:
                    arr = arr.reshape(-1, 1)
            elif arr.ndim == 2:
                pass
            else:
                arr = arr.reshape(arr.shape[0], -1)
            proc.append(arr)
        angle_arr = np.concatenate(proc, axis=0)
        # common accidental transpose: (11,1) -> (1,11)
        if angle_arr.shape[1] == 1 and angle_arr.shape[0] == 11:
            angle_arr = angle_arr.T

    if angle_arr.ndim != 2 or angle_arr.shape[1] != 11:
        sample_shapes = [np.asarray(x).shape for x in angle_results[:3]]
        raise ValueError(
            f"compute_angles produced unexpected shape {angle_arr.shape}; expected (N,11). sample_shapes={sample_shapes}"
        )

    print(f"Computed angle array shape: {angle_arr.shape} for model {model_id}")
    angle_results = pd.DataFrame(data=angle_arr, columns=ANGLE_COLUMNS)
    angle_results = pd.concat([data_df, angle_results], axis=1)
    angle_results.to_excel(
        out_dir / f"model_{model_id}.xlsx",
        index=False,
        sheet_name="Union",
    )

    # Delete the model and free the gpu memory. Required for LLMs
    if hasattr(model, 'model'):
        model.model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"Model: {model_id}, Time Taken: {time.process_time() - model_time:.2f} s"
    )

    print("Done")


if __name__ == "__main__":
    program_time = time.process_time()
    main()
    print(f"Done! Time Taken: {time.process_time() - program_time:.2f} s")
