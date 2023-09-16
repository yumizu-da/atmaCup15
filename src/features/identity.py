from typing import List, Union

import pandas as pd

from src.features.base import AbstractBaseBlock


class IdentityBlock(AbstractBaseBlock):
    """特徴量をそのまま返すBlock

    Args:
        columns (Union[str, List[str]]): カラム名
    """

    def __init__(self, columns: Union[str, List[str]]):
        self.columns = columns

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """特徴量をそのまま返す

        Args:
            input_df (pd.DataFrame): 説明変数

        Returns:
            pd.DataFrame: カラムをそのまま返す
        """
        return input_df[self.columns]
