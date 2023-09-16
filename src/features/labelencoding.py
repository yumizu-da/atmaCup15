from typing import List, Optional, Union

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.features.base import AbstractBaseBlock


class LabelEncodingBlock(AbstractBaseBlock):
    """ラベルエンコーディングを行う"""

    def __init__(self, columns: Union[str, List[str]], source_df: Optional[pd.DataFrame] = None):
        self.columns = columns
        self.source_df = source_df

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """学習用データに対し, ラベルエンコーディングを行う

        未知の値は-1に変換. NanはNanのままにする
        """
        self.oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        if self.source_df is not None:
            self.oe.fit(self.source_df[self.columns])
        else:
            self.oe.fit(input_df[self.columns])

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """学習時にfitしたラベルエンコーディングを行う"""
        output_df = input_df[self.columns].copy()
        output_df = self.oe.transform(output_df)
        output_df = pd.DataFrame(output_df, columns=self.columns)

        return output_df.add_prefix("le_").astype(int).astype("category")
