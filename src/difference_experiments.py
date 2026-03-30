import os
from collections import defaultdict
import gc
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

from utils import get_arguments, DF, METRICS_MAPPING, mkdir_p, derive_model_label

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
try:
    from Models.llm_embeddings import build_embedder
    print("Successfully imported build_embedder.")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

DIFFERENCE_SENTENCES_COLUMNS = ["A", "B", "D"]


def load_difference_data(sheet: str) -> DF:
    data_pt = Path("./data/difference_analysis.xlsx")

    nrows = 128 if sys.gettrace() is not None else None

    return pd.read_excel(
        data_pt,
        sheet_name=sheet,
        usecols=DIFFERENCE_SENTENCES_COLUMNS,
        nrows=nrows,
    )


def encode_chunk(model, chunk: pd.DataFrame, encode_batch_size: int):
    """Encode A, B, D in a single model call."""
    a = chunk["A"].tolist()
    b = chunk["B"].tolist()
    d = chunk["D"].tolist()

    all_texts = a + b + d

    with torch.inference_mode():
        all_embeds = model.encode(all_texts, batch_size=encode_batch_size)

    # Convert to numpy once if needed
    if isinstance(all_embeds, torch.Tensor):
        all_embeds = all_embeds.detach().cpu().numpy()
    else:
        all_embeds = np.asarray(all_embeds)

    n = len(chunk)
    a_emb = all_embeds[:n]
    b_emb = all_embeds[n:2 * n]
    d_emb = all_embeds[2 * n:3 * n]

    return {
        "A": a_emb,
        "B": b_emb,
        "D": d_emb,
        "(A-B)": a_emb - b_emb,
    }


def compute_metrics_for_chunk(chunk_embeds, metrics_map, model_id):
    """Compute all requested pairwise metrics for one chunk."""
    results = defaultdict(list)

    all_pairs = [
        ("A", "D"),
        ("A", "B"),
        ("B", "D"),
        ("(A-B)", "D"),
        ("(A-B)", "B"),
    ]

    for metric_id, metric_fn in metrics_map.items():
        for left_key, right_key in all_pairs:
            pair_id = f"{left_key}_{right_key}"
            metric_vals = metric_fn(chunk_embeds[left_key], chunk_embeds[right_key])

            # Make sure each result is 1D for column creation
            metric_vals = np.asarray(metric_vals).reshape(-1)
            results[f"{model_id}_{metric_id}_{pair_id}"].append(metric_vals)

    return results


def analyze(diff_df: DF, data_key: str, model, model_id: str, out_dir: Path,
            metrics_map, chunk_size: int, encode_batch_size: int, write_mode: str = "w") -> None:
    assert write_mode in {"w", "a"}

    print(f"\nModel: {model_id}, Data Key: {data_key}")
    total_start = time.perf_counter()

    all_results = defaultdict(list)

    n_rows = len(diff_df)
    n_chunks = max((n_rows + chunk_size - 1) // chunk_size, 1)

    for chunk_idx, start in enumerate(range(0, n_rows, chunk_size)):
        chunk_start = time.perf_counter()
        chunk = diff_df.iloc[start:start + chunk_size]

        chunk_embeds = encode_chunk(model, chunk, encode_batch_size)
        chunk_results = compute_metrics_for_chunk(chunk_embeds, metrics_map, model_id)

        for key, value_list in chunk_results.items():
            all_results[key].extend(value_list)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - chunk_start
        print(f"Chunk {chunk_idx + 1}/{n_chunks} done in {elapsed:.2f}s")

    # Flatten results into output columns
    out_dict = {}
    for key, value_list in all_results.items():
        out_dict[key] = np.concatenate(value_list, axis=0)

    result_df = pd.DataFrame(out_dict)
    out = pd.concat([diff_df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)

    output_path = out_dir / f"{model_id}" / "C2_and_C3_results.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, mode=write_mode, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name=data_key)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_elapsed = time.perf_counter() - total_start
    print(f"Model: {model_id}, Data Key: {data_key}, Time Taken: {total_elapsed:.2f} s")


def main():
    args = get_arguments()
    exp_id = "difference"
    out_dir = mkdir_p(Path(args.output_dir).joinpath(f"{exp_id}_results"))

    # Separate chunk_size from encode batch size
    encode_batch_size = args.batch_size
    chunk_size = getattr(args, "chunk_size", max(256, encode_batch_size * 8))

    if sys.gettrace() is not None:
        out_dir = mkdir_p(Path("../temp/").joinpath(f"{exp_id}_results"))
        encode_batch_size = 1
        chunk_size = 16

    metrics_map = METRICS_MAPPING

    if sys.gettrace() is not None:
        model_name = "Qwen/Qwen3-Embedding-8B"
        model = build_embedder(model_name=model_name)
        print("Debug mode: Using Qwen3-Embedding-8B model for faster testing.")
    else:
        model_name = args.model
        model = build_embedder(model_name=model_name, device=args.gpu)

    model_id = derive_model_label(model_name, None)

    ldiff_data = load_difference_data(sheet="LDiff")
    analyze(
        diff_df=ldiff_data,
        data_key="LDiff",
        model=model,
        model_id=model_id,
        out_dir=out_dir,
        metrics_map=metrics_map,
        chunk_size=chunk_size,
        encode_batch_size=encode_batch_size,
        write_mode="w",
    )

    rdiff_data = load_difference_data(sheet="RDiff")
    analyze(
        diff_df=rdiff_data,
        data_key="RDiff",
        model=model,
        model_id=model_id,
        out_dir=out_dir,
        metrics_map=metrics_map,
        chunk_size=chunk_size,
        encode_batch_size=encode_batch_size,
        write_mode="a",
    )

    # Cleanup
    if hasattr(model, "model"):
        model.model.cpu()

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Done")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Done. Time Taken: {time.perf_counter() - start:.2f} s")