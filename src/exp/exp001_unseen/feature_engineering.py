import hashlib
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from gensim.models import word2vec
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.features.base import AbstractBaseBlock
from src.features.run_blocks import run_blocks
from src.utils.utils import seed_everything


class OnehotAnimeByUserBlock(AbstractBaseBlock):
    """各ユーザーに対して視聴したアニメのonehotベクトルを作成"""

    def __init__(self, source_df: pd.DataFrame):
        self.source_df: pd.DataFrame = source_df

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        self.master_df = pd.crosstab(self.source_df["user_id"], self.source_df["anime_id"]).reset_index()
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.merge(input_df[["user_id"]], self.master_df, on="user_id", how="left")
        output_df = output_df.drop("user_id", axis=1)
        output_df = output_df.add_prefix("onehot_anime_by_user_")
        return output_df


class OnehotUserByAnimeBlock(AbstractBaseBlock):
    """各アニメに対して視聴したユーザーのonehotベクトルを作成"""

    def __init__(self, source_df: pd.DataFrame):
        self.source_df: pd.DataFrame = source_df

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        self.master_df = pd.crosstab(self.source_df["anime_id"], self.source_df["user_id"]).reset_index()
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.merge(input_df[["anime_id"]], self.master_df, on="anime_id", how="left")
        output_df = output_df.drop("anime_id", axis=1)
        output_df = output_df.add_prefix("onehot_user_by_anime_")
        return output_df


class Word2vecAnimeBlock(AbstractBaseBlock):
    """各ユーザーが評価したアニメ一覧に対しWord2vecを実行"""

    def __init__(self, groupby_col: str, target_col: str, source_df: pd.DataFrame, vector_size: int = 64):
        self.groupby_col: str = groupby_col
        self.target_col: str = target_col
        self.source_df: pd.DataFrame = source_df
        self.vector_size: int = vector_size

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:

        group_df = self.source_df.groupby(self.groupby_col)[self.target_col].apply(list).reset_index()
        w2v_model = word2vec.Word2Vec(
            group_df[self.target_col].values.tolist(),
            vector_size=self.vector_size,
            min_count=1,
            window=256,
            epochs=50,
            seed=42,
            workers=1,
            hashfxn=self._hashfxn,
        )
        sentence_vectors = group_df[self.target_col].apply(lambda x: np.mean([w2v_model.wv[e] for e in x], axis=0))
        sentence_vectors = np.vstack([x for x in sentence_vectors])
        sentence_vector_df = pd.DataFrame(
            sentence_vectors,
            columns=[f"{self.target_col}_w2v_{i}" for i in range(self.vector_size)],
        )
        sentence_vector_df.index = group_df[self.groupby_col]
        self.sentence_vector_df = sentence_vector_df.reset_index()

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.merge(
            input_df[self.groupby_col],
            self.sentence_vector_df,
            on=self.groupby_col,
            how="left",
        )
        output_df.drop(self.groupby_col, axis=1, inplace=True)

        return output_df

    def _hashfxn(self, x):
        """Word2vecのseed固定用"""
        return int(hashlib.md5(str(x).encode()).hexdigest(), 16)


class Word2vecUserBlock(AbstractBaseBlock):
    """各アニメを視聴したユーザー一覧に対しWord2vecを実行"""

    def __init__(self, groupby_col: str, target_col: str, source_df: pd.DataFrame, vector_size: int = 64):
        self.groupby_col: str = groupby_col
        self.target_col: str = target_col
        self.source_df: pd.DataFrame = source_df
        self.vector_size: int = vector_size

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:

        group_df = self.source_df.groupby(self.groupby_col)[self.target_col].apply(list).reset_index()
        w2v_model = word2vec.Word2Vec(
            group_df[self.target_col].values.tolist(),
            vector_size=self.vector_size,
            min_count=1,
            window=256,
            epochs=50,
            seed=42,
            workers=1,
            hashfxn=self._hashfxn,
        )
        sentence_vectors = group_df[self.target_col].apply(lambda x: np.mean([w2v_model.wv[e] for e in x], axis=0))
        sentence_vectors = np.vstack([x for x in sentence_vectors])
        sentence_vector_df = pd.DataFrame(
            sentence_vectors,
            columns=[f"{self.target_col}_w2v_{i}" for i in range(self.vector_size)],
        )
        sentence_vector_df.index = group_df[self.groupby_col]
        self.sentence_vector_df = sentence_vector_df.reset_index()

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.merge(
            input_df[self.groupby_col],
            self.sentence_vector_df,
            on=self.groupby_col,
            how="left",
        )
        output_df.drop(self.groupby_col, axis=1, inplace=True)

        return output_df

    def _hashfxn(self, x):
        """Word2vecのseed固定用"""
        return int(hashlib.md5(str(x).encode()).hexdigest(), 16)


class GroupbyBlock(AbstractBaseBlock):
    """use_idをkeyにしてgroupbyしたanimeに関数特徴量を作成"""

    def __init__(
        self,
        columns: List[str],
        groupby_column: str,
        source_df: Optional[pd.DataFrame] = None,
        agg_funcs: List[str] = ["mean", "std", "max", "min"],
    ):
        self.columns = columns
        self.groupby_column = groupby_column
        self.source_df = source_df
        self.agg_funcs = agg_funcs

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:

        if self.source_df is not None:
            self.agg_df = self.source_df.copy()
        else:
            self.agg_df = input_df.copy()

        dfs: List[pd.DataFrame] = []
        for c in self.columns:
            groupby_df = (
                self.agg_df.groupby(self.groupby_column)
                .agg({c: self.agg_funcs})
                .reset_index()
                .set_index(self.groupby_column)
            )
            groupby_df.columns = [f"groupby_{self.groupby_column}_{c}_{agg_func}" for agg_func in self.agg_funcs]
            dfs.append(groupby_df)
        self.group_df = pd.concat(dfs, axis=1).reset_index()

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.merge(input_df[self.groupby_column], self.group_df, on=self.groupby_column, how="left")
        output_df = output_df.drop(self.groupby_column, axis=1).reset_index(drop=True)

        return output_df


if __name__ == "__main__":

    # このファイルがあるパスより, 実験番号(ex. exp001)及びconfig.yamlのパスを取得
    abs_path: str = os.path.dirname(os.path.abspath(__file__))
    exp_number: str = abs_path.split("/")[-1]
    config_path: Path = Path(abs_path, "config.yaml")

    config: Union[DictConfig, ListConfig] = OmegaConf.load(config_path)
    seed_everything(config.seed)

    train_df = pd.read_csv(config.train_csv_path)
    test_df = pd.read_csv(config.test_csv_path)
    anime_df = pd.read_csv(config.anime_csv_path)
    train_df = pd.merge(train_df, anime_df, on="anime_id", how="left")
    test_df = pd.merge(test_df, anime_df, on="anime_id", how="left")
    all_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    # NOTE: Unseenデータ
    test_df = test_df[~test_df["user_id"].isin(train_df["user_id"])].reset_index(drop=True)

    # 特徴量定義
    feature_blocks: List[AbstractBaseBlock] = [
        Word2vecAnimeBlock("user_id", "anime_id", all_df),
        Word2vecUserBlock("anime_id", "user_id", all_df),
        GroupbyBlock(["members", "watching", "completed", "on_hold", "dropped", "plan_to_watch"], "user_id", all_df),
    ]

    # 特徴量作成
    save_dir: Path = Path(config.features_dir, exp_number)
    run_blocks(train_df, feature_blocks, save_dir, is_train=True)
    run_blocks(test_df, feature_blocks, save_dir, is_train=False)

    # 目的変数保存
    target_train_df: pd.DataFrame = pd.DataFrame(train_df[config.target_col])
    target_train_df.to_pickle(Path(config.features_dir, exp_number, config.target_train_file))

    # GroupKfoldを使用する場合はgroup_colも保存
    if config.group_col is not None:
        group_train_df: pd.DataFrame = pd.DataFrame(train_df[config.group_col])
        group_train_df.to_pickle(Path(config.features_dir, exp_number, config.group_train_file))
