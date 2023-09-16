import os
from pathlib import Path
from typing import List, Union

import pandas as pd

from src.features import AbstractBaseBlock
from src.utils.timer import timer


def run_blocks(
    input_df: pd.DataFrame,
    blocks: List[AbstractBaseBlock],
    save_dir: Union[str, Path],
    is_train: bool = False,
) -> None:
    """特徴量エンジニアリングを実施"""
    print("🚀 Start feature engineering")

    mode: str = "train" if is_train else "test"

    for block in blocks:
        with timer(f"{block.__class__.__name__}"):
            if is_train:
                _df = block.fit(input_df)
            else:
                _df = block.transform(input_df)

        assert len(_df) == len(input_df), block

        os.makedirs(save_dir, exist_ok=True)
        _df.to_pickle(Path(save_dir, f"{block.__class__.__name__}_{mode}.pkl"))

    print("🎉 Finished feature engineering")
