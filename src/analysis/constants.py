from typing import List

import pandas as pd

DF = pd.DataFrame

MODELS: List[str] = ["gpt", "llama2", "lsr", "sbert", "use"]  # lsr -> Laser

MODEL_LIST = {
    "old": ["gpt", "llama2", "lsr", "sbert", "use"],
    "classical": ["roberta", "mpnet", "simcse", "infersent"],
    "llm": ["mistral", "llama3", "olmo", "openelm"],
    "llm_dec": ["llama32", "gemma", "qwen"],
    "2026": ["qwen3embedding", "mistrale5", "octenaembedding"],
}

MODEL_FILE_LIST = {
    "old": "",
    "classical": "model_",
    "llm": "model_",
    "llm_dec": "model_",
    "2026": "model_",
}

COLOR_LIST = {
    "old": [
        (214, 117, 111),  # Coral Red
        (133, 187, 120),  # Soft Green
        (244, 170, 114),  # Peach Orange
        (117, 158, 199),  # Sky Blue
        (164, 136, 198),  # Lavender Purple,
    ],
    "classical": [
        (198, 151, 71),  # Mustard Yellow
        (43, 74, 107),  # Navy Blue
        (238, 182, 193),  # Blush Pink
        (85, 87, 89),  # Charcoal Gray
    ],
    "llm": [
        (64, 224, 208),  # Turquoise
        (204, 85, 0),  # Burnt Orange
        (152, 255, 152),  # Mint Green
        (75, 0, 130),  # Deep Purple
        (255, 223, 186),  # Soft Gold
    ],
    "llm_dec": [
        (0, 128, 128),  # Teal
        (255, 255, 0),  # Yellow
        (128, 0, 128),  # Purple
    ],
    "2026": [
    (0, 200, 83),    # Emerald Green
    (255, 45, 85),   # Vivid Crimson
    (0, 122, 255),  ] # Bright Azure Blue
}


def remove_common_rows_from_df(df1: DF, df2: DF) -> DF:
    """Removes the rows from df1 that are present in df2
    Source - https://stackoverflow.com/a/33282617
    """
    keys = list(df2.columns.values)
    i1 = df1.set_index(keys).index
    i2 = df2.set_index(keys).index
    return df1[~i1.isin(i2)]