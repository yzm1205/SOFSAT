import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from classical_encoders import MODEL_ENCODER_MAPPING
from utils import get_arguments, DF, METRICS_MAPPING, mkdir_p, compute_angles

import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
    print("current_device =", torch.cuda.current_device())

UNION_SENTENCES_COLUMNS = ["S1", "S2", "Sy"]

# Filter Some Samples
FLAG_COL = "duplicate"


# TODO: Yash, you can probably put this part in utils so that it can be reused since there'll be duplicates
def read_final_combined_data():
    pt = Path("./data/final_combined_data.xlsx")
    return pd.read_excel(pt)


def get_gold_union_data() -> DF:
    data_df = read_final_combined_data()
    data_df = data_df[data_df[FLAG_COL] == 0]
    union1 = data_df[["previous", "S0", "S1"]].rename(
        columns={"previous": "S1", "S0": "S2", "S1": "Sy"}
    )
    union2 = data_df[["S0", "next", "S2"]].rename(
        columns={"S0": "S1", "next": "S2", "S2": "Sy"}
    )
    union_data = pd.concat([union1, union2], axis=0, ignore_index=True)

    return union_data


def filter_union_duplicates(df: DF) -> DF:
    cols = ["S1", "S2", "Sy"]
    df = df.drop_duplicates(subset=cols)

    final_combined_data = get_gold_union_data()

    out = df.merge(final_combined_data, on=cols, how="inner", indicator=True)
    out = out[out["_merge"] == "both"]
    out = out.drop("_merge", axis=1)

    assert len(out) == len(final_combined_data)

    print(f"Length of original DF: {len(df)}, Length of filtered DF: {len(out)} ")

    return out


def load_union_data() -> DF:
    """Load the data corresponding to overlap. Paths are hardcoded"""

    data_pt = Path(
        "./data/LocationExp/use.xlsx"
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
        skiprows=range(5),
    )

    return data


def main():
    args = get_arguments()

    exp_id = "union"
    out_dir = mkdir_p(Path(args.output_dir).joinpath(f"{exp_id}_results"))

    if sys.gettrace() is not None:  # Debug
        # args.model = "infersent"
        args.batch_size = 2048
        out_dir = mkdir_p(Path("../temp/").joinpath(f"{exp_id}_results"))

    # Data
    data_df = load_union_data()

    # Filter where S1 and S2 are same.
    data_df = filter_union_duplicates(data_df)

    n_chunks = len(data_df) // args.batch_size
    n_chunks = 1 if not n_chunks else n_chunks

    # Metrics
    metrics_map = METRICS_MAPPING

    # Iterate over all the models
    for model_id, model_cls in MODEL_ENCODER_MAPPING.items():
        print(f"\nModel: {model_id}")
        
        # if model_id == "qwen3embedding":
        #     print("================================================\n")
        #     print("Skipping Qwen3Embedding for overlap experiments")
        #     print("================================================\n")
            
        #     continue
        model_time = time.process_time()
        # Load model
        model = model_cls(device=True)

        # Iterate over batch of data and generate results
        angle_results = []
        for b_idx, chunk in enumerate(np.array_split(data_df, n_chunks)):
            batch_time = time.process_time()

            # Encode S0, S1 and S2 for data chunk
            chunk_embeds = {
                key: model.encode(chunk[key].to_list(), batch_size=args.batch_size)
                for key in UNION_SENTENCES_COLUMNS
            }

            # Sanity-check embeddings: must be 2D and have the same embedding-dim
            embed_shapes = {k: np.asarray(v).shape for k, v in chunk_embeds.items()}
            print(f"DEBUG embed shapes (model={model_id} batch={b_idx}): {embed_shapes}")
            dims = {s[1] for s in embed_shapes.values() if len(s) == 2}
            if not all(len(s) == 2 for s in embed_shapes.values()) or len(dims) != 1:
                raise ValueError(f"Unexpected embedding shapes from model '{model_id}' for keys {list(chunk_embeds.keys())}: {embed_shapes}")

            # Generate the projection results
            angle_results.append(
                compute_angles(
                    chunk_embeds["Sy"], chunk_embeds["S1"], chunk_embeds["S2"]
                )
            )

            print(
                f"Model: {model_id}, Batch_Idx: {b_idx}/{n_chunks - 1} Time: {time.process_time() - batch_time}"
            )

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
        angle_results = pd.DataFrame(
            data=angle_arr,
            columns=[
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
            ],
        )
        angle_results = pd.concat([data_df, angle_results], axis=1)
        angle_results.to_excel(
            mkdir_p(out_dir.joinpath(f"model_{model_id}.xlsx")),
            index=False,
            sheet_name="Union",
        )
        print(
            f"Model: {model_id}, Time Taken: {time.process_time() - model_time:.2f} s"
        )

        # Delete the model and free the gpu memory
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("Done")


if __name__ == "__main__":
    program_time = time.process_time()
    main()
    print(f"Done! Time Taken: {time.process_time() - program_time:.2f} s")
