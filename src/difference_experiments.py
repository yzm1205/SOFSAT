import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # keep only GPU 3 visible to this process
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "3")
# os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

from collections import defaultdict
import gc
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


from utils import get_arguments, DF, METRICS_MAPPING, mkdir_p, compute_angles
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
try:
    from Models.llm_embeddings import build_embedder, _derive_model_label
    print("Successfully imported build_embedder.")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


DIFFERENCE_SENTENCES_COLUMNS = ["A", "B", "D"]


def load_difference_data(sheet: str) -> DF:
    """Load the data corresponding to overlap. Paths are hardcoded"""
    data_pt = Path(
        "./data/difference_analysis.xlsx"
    )

    # Read small number of rows in debug mode
    nrows = None
    if sys.gettrace() is not None:  # Debug
        nrows = 128

    data = pd.read_excel(
        data_pt, sheet_name=sheet, usecols=DIFFERENCE_SENTENCES_COLUMNS, nrows=nrows
    )

    return data


def main():
    args = get_arguments()
    exp_id = "difference"
    out_dir = mkdir_p(Path(args.output_dir).joinpath(f"{exp_id}_results"))

    if sys.gettrace() is not None:  # Debug
        out_dir = mkdir_p(Path("../temp/").joinpath(f"{exp_id}_results"))
        args.batch_size = 1

    # Metrics
    metrics_map = METRICS_MAPPING
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

    def analyze(
        diff_df: DF, data_key: str, n_chunks: int, write_mode: str = "w"
    ) -> None:
        # Iterate over all the models
        assert write_mode in ["w", "a"]
        model_results = [
            diff_df,
        ]
        
        # for model_id, model_cls in MODEL_ENCODER_MAPPING.items():
        print(f"\nModel: {model_id}, Data Key: {data_key}")
        model_time = time.process_time()
    
        # Iterate over batch of data and generate results
        results = defaultdict(list)
        angle_results = []
        for b_idx, chunk in enumerate(np.array_split(diff_df, n_chunks)):
            batch_time = time.process_time()
            # Encode S0, S1 and S2 for data chunk
            chunk_embeds = {
                key: model.encode(chunk[key].to_list(), batch_size=args.batch_size)
                for key in DIFFERENCE_SENTENCES_COLUMNS
            }
            chunk_embeds["(A-B)"] = chunk_embeds["A"] - chunk_embeds["B"]

            # TODO: Here
            # Compute all metric results for [(A, D), (A, B), (B, D), ((A-B), D), ((A-B), B)]
            all_pairs = [
                ("A", "D"),
                ("A", "B"),
                ("B", "D"),
                ("(A-B)", "D"),
                ("(A-B)", "B"),
            ]
            for metric_id, metric_cls in metrics_map.items():
                for pair_embeds in all_pairs:
                    pair_id = "_".join(pair_embeds)
                    metric_inputs = [
                        chunk_embeds[p_embed] for p_embed in pair_embeds
                    ]
                    results[f"{model_id}_{metric_id}_{pair_id}"].append(
                        metric_cls(*metric_inputs)
                    )

            # Generate the projection results
            angle_results.append(
                compute_angles(
                    chunk_embeds["D"], chunk_embeds["A"], chunk_embeds["B"]
                )
            )

            print(
                f"Model: {model_id}, Data Key: {data_key}, Batch_Idx: {b_idx}/{n_chunks -1} Time: {time.process_time() - batch_time}"
            )

        # H1 Results per model
        out = {}
        for key, val in results.items():
            out[key] = np.concatenate(val).squeeze()
        model_results.append(pd.DataFrame.from_dict(out))
        print(
            f"Model: {model_id}, Time Taken: {time.process_time() - model_time:.2f} s"
        )

        # H2/Angle Results per model
        # angle_results = pd.DataFrame(
        #     data=np.concatenate(angle_results, axis=0),
        #     columns=[
        #         "Projection Location",
        #         "Angle_A_B",
        #         "Angle_A_Projection",
        #         "Angle_B_Projection",
        #         "Norm_A",
        #         "Norm_B",
        #         "Norm_Proj",
        #         "Norm_Orig_Vec",
        #         "TS_Proj_A",
        #         "TS_Proj_B",
        #         "TS_A_B",
        #     ],
        # )
        # angle_results = pd.concat([diff_df, angle_results], axis=1)

        # with pd.ExcelWriter(
        #     mkdir_p(out_dir.joinpath(f"C2_results/model_{model_id}.xlsx")),
        #     mode=write_mode,
        # ) as writer:
        #     angle_results.to_excel(
        #         writer,
        #         index=False,
        #         sheet_name=data_key,
        #     )

        # H1 Results Across model
        out = pd.concat(model_results, axis=1)

        with pd.ExcelWriter(
            mkdir_p(out_dir.joinpath(f"C2_and_C3_results.xlsx")),
            mode=write_mode,
        ) as writer:
            out.to_excel(
                writer,
                index=False,
                sheet_name=data_key,
            )

    # Analyze Left Difference
    sheet = "LDiff"
    ldiff_data = load_difference_data(sheet=sheet)
    chunk_ct = len(ldiff_data) // args.batch_size
    chunk_ct = chunk_ct if chunk_ct > 0 else 1
    analyze(ldiff_data, data_key=sheet, n_chunks=chunk_ct, write_mode="w")

    # Analyze Right Difference Data
    sheet = "RDiff"
    rdiff_data = load_difference_data(sheet=sheet)
    chunk_ct = len(rdiff_data) // args.batch_size
    chunk_ct = chunk_ct if chunk_ct > 0 else 1
    analyze(rdiff_data, data_key=sheet, n_chunks=chunk_ct, write_mode="a")

    # Clean up model and free GPU memory after all analysis is done
    if hasattr(model, 'model'):
        model.model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("Done")


if __name__ == "__main__":
    program_time = time.process_time()
    main()
    print(f"Done. Time Taken: {time.process_time() - program_time:.2f} s")
