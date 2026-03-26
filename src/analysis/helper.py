import sys
from pathlib import Path
import string
import time
from typing import Tuple

import pandas as pd

from utils import DF, mkdir_p


INTERSECTION_COLS = ["S0", "S1", "S2"]
FLAG_COL = "duplicate"


def _load_full_data() -> DF:
    """Orig data that Sanjeev gave me. It had 37357 samples.
    Even if I remove duplicates across intersection data, I am left with 37304 samples.
    Thus, I have to take overlap with filtered data to get the same amount."""
    pt = Path("/data/naman/test_auto_encoder/data/cnn_in_chatgpt_fusion_out.xlsx")
    full_data = pd.read_excel(pt)
    full_data = full_data.dropna()
    full_data = full_data.drop_duplicates(subset=INTERSECTION_COLS, ignore_index=True)
    return full_data


def _load_intersection_analysis_data() -> DF:
    """Here Sanjeev did some filtration to remove duplicates. This gives me 37301 samples"""
    data = pd.read_excel(
        "/data/naman/test_auto_encoder/Results/SanjeevPaper/intersection_analysis.xlsx",
        usecols=INTERSECTION_COLS,
    )
    assert len(data.columns) == len(INTERSECTION_COLS)
    return data


def load_synthetic_data() -> DF:
    full_data = _load_full_data()
    filtered_intersection_data = _load_intersection_analysis_data()
    merged_full_data = pd.merge(
        full_data, filtered_intersection_data, on=INTERSECTION_COLS
    )  # This is done to get prev and next cols as well
    assert len(merged_full_data) == len(filtered_intersection_data)
    return merged_full_data


def remove_punctuations(s: str) -> str:
    """Removes punctuation and spaces from a string"""
    return "".join(filter(str.isalnum, s))


def _find_consecutive_duplicates(row):
    prev = remove_punctuations(row.previous)
    curr = remove_punctuations(row.S0)
    nxt = remove_punctuations(row.next)
    if prev == curr or curr == nxt:
        return 1
    return 0


def save_final_combined_data(pt: Path) -> None:
    # Combined data after removing duplicates
    combined_data = load_synthetic_data()
    combined_data.to_excel(mkdir_p(pt), index=False)


def flag_consecutive_duplicates(combined_data_pt: Path, save_pt: Path):
    combined_data = pd.read_excel(combined_data_pt)

    # In some cases, consecutive sentences are same. Flag those
    combined_data[FLAG_COL] = combined_data.apply(_find_consecutive_duplicates, axis=1)
    combined_data.to_excel(mkdir_p(save_pt), index=False)


def save_removed_samples():
    pt = Path("/data/naman/test_auto_encoder/data/cnn_in_chatgpt_fusion_out.xlsx")
    full_data = pd.read_excel(pt)
    full_data = full_data.drop_duplicates(ignore_index=True)

    # final combined data
    save_pt = Path("/data/naman/test_auto_encoder/data/final_combined_data.xlsx")
    final_data = pd.read_excel(save_pt)
    final_data = final_data[final_data[FLAG_COL] == 0]
    final_data = final_data[full_data.columns]

    # Save the remaining ones
    removed_samples = pd.concat([full_data, final_data]).drop_duplicates(keep=False)
    assert len(removed_samples) == len(full_data) - len(final_data)
    removed_samples.to_excel(
        "/data/naman/test_auto_encoder/data/final_removed_data.xlsx", index=False
    )
    aa = 1


def save_difference_samples_for_removed_samples():
    def filter_df(in_data: DF, keys: Tuple[str, str, str]):
        out = in_data[list(keys)].drop_duplicates()
        return out.rename(columns={keys[0]: "A", keys[1]: "B", keys[2]: "D"})

    data = pd.read_excel("/data/naman/test_auto_encoder/data/final_removed_data.xlsx")

    # LDiff
    out = [
        filter_df(data, ("S1", "previous", "S0")),
        filter_df(data, ("S1", "S0", "previous")),
        filter_df(data, ("S1", "S2", "previous")),
    ]
    out = pd.concat(out, axis=0)
    out.to_excel(
        "/data/naman/test_auto_encoder/data/final_removed_left_difference_data.xlsx",
        index=False,
    )

    # RDiff
    out = [
        filter_df(data, ("S2", "next", "S0")),
        filter_df(data, ("S2", "S0", "next")),
        filter_df(data, ("S2", "S1", "next")),
    ]
    out = pd.concat(out, axis=0)
    out.to_excel(
        "/data/naman/test_auto_encoder/data/final_removed_right_difference_data.xlsx",
        index=False,
    )


def main():
    # Note:  This is to save the final data (prev, curr, next, S1, S2)
    # pt = Path("/data/naman/test_auto_encoder/data/combined_data.xlsx")
    #
    # # Save Combined Data
    # save_final_combined_data(pt)
    #
    # # Do extra filtration and save
    # save_pt = Path("/data/naman/test_auto_encoder/data/final_combined_data.xlsx")
    # flag_consecutive_duplicates(pt, save_pt)

    # # Note: Save Removed Samples after filtration. This is done for filtering the Difference samples
    # save_removed_samples()

    # # Note: Now find the difference samples corresponding to the samples that have been removed
    save_difference_samples_for_removed_samples()


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    print(f"Time Taken: {time.perf_counter() - start_time}")
