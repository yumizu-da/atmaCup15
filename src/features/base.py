from abc import ABCMeta, abstractmethod

import pandas as pd


class AbstractBaseBlock(metaclass=ABCMeta):
    """特徴量変換の基底クラス

    Args:
        input_df (pd.DataFrame): 説明変数

    Raises:
        NotImplementedError: 子クラスでtransformメソッドを実装していない場合に発生
    """

    def fit(self, input_df: pd.DataFrame):
        """学習用データに対し, 内部状態の更新及び特徴量変換を行う"""
        return self.transform(input_df)

    @abstractmethod
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """特徴量変換を実施

        Args:
            input_df (pd.DataFrame): 説明変数

        Returns:
            pd.DataFrame: 特徴量変換した説明変数
        """
        raise NotImplementedError()
