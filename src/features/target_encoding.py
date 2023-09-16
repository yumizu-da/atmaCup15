import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.features.base import AbstractBaseBlock


class TargetEncodingBlock(AbstractBaseBlock):
    """ターゲットエンコーディングを行うBlock"""

    def __init__(self, column: str, target_column: str):
        self.column = column
        self.target_column = target_column

    def fit(self, input_df: pd.DataFrame):
        self.target_map = input_df.groupby(self.column)[self.target_column].mean()

        output_encoded = np.zeros(len(input_df))
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        x = input_df.drop(self.target_column, axis=1)
        y = input_df[self.target_column].values

        for train_idx, valid_idx in cv.split(x, y):
            _map = input_df.iloc[train_idx].groupby(self.column)[self.target_column].mean().to_dict()
            output_encoded[valid_idx] = input_df.iloc[valid_idx][self.column].map(_map)

        output_df = pd.DataFrame(output_encoded).add_prefix("te_")
        return output_df

    def transform(self, input_df: pd.DataFrame):
        output_encoded = input_df[self.column].map(self.target_map).values
        output_df = pd.DataFrame(output_encoded).add_prefix("te_")
        return output_df
