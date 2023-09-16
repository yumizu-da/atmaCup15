import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytz import timezone  # type: ignore

from src.models.select_model import select_model
from src.utils.metric import metric
from src.utils.select_cv_method import select_cv_method
from src.utils.visualizer import create_feature_importance


class Runner:
    """学習/推論を行うクラス"""

    def __init__(self, config: Union[DictConfig, ListConfig], exp_number: str, date_str: str):
        self.config = config
        self.exp_number = exp_number
        self.date_str = date_str
        self.model_cls, self.params, self.fit_params = select_model(config)
        self.group = None

    def train_cv(self) -> None:
        """CVを用いた学習"""

        # 学習データの読込
        x_train = self.load_x_train()
        y_train = self.load_y_train()

        # CVの設定
        cv = select_cv_method(self.config.cv_method, self.config.n_splits, self.config.shuffle, self.config.seed)
        group = self.load_group()

        oof = np.zeros(len(x_train))
        scores: List[float] = []
        feature_importances_list: List[pd.DataFrame] = []

        for i_fold, (train_idx, valid_idx) in enumerate(cv.split(x_train, y_train, group)):

            run_name = self.get_run_name(i_fold)
            logger.info(f"🚀 Start training: {run_name}")

            tr_x, tr_y = x_train.iloc[train_idx], y_train.iloc[train_idx]
            va_x, va_y = x_train.iloc[valid_idx], y_train.iloc[valid_idx]

            # 学習
            model = self.model_cls(run_name, self.params, self.fit_params)
            model.fit(tr_x, tr_y, va_x, va_y)
            model.save_model(self.model_dir)

            # 推論
            tr_pred = model.predict(tr_x)
            va_pred = model.predict(va_x)
            tr_score = metric(self.config.metric, tr_y, tr_pred)
            va_score = metric(self.config.metric, va_y, va_pred)
            logger.info(
                f"[Score] train's {self.config.metric}: {tr_score:.4f} valid's {self.config.metric}: {va_score:.4f}"
            )

            # 結果を保持
            oof[valid_idx] = va_pred
            scores.append(va_score)

            # TODO: この分岐をもっとスマートに書きたい
            if self.config.model_name == "LightGBM":
                feature_importance = model.feature_importance(tr_x)
                feature_importances_list.append(feature_importance)

        # CVスコアを計算
        cv_score = metric(self.config.metric, y_train, oof)
        logger.info(f"CV: {cv_score:.4f}")

        # TODO: この分岐をもっとスマートに書きたい
        if self.config.model_name == "LightGBM":
            fig = create_feature_importance(feature_importances_list)
            self.save_fig(fig)

        # oofを保存
        self.save_oof(oof)

        logger.info("🎉 Finish training!!")

    def predict_cv(self) -> None:
        """学習で作成したモデルを用いて推論を行う"""

        logger.info("🦖 start prediction cv")

        x_test = self.load_x_test()

        preds = []
        for i_fold in range(self.config.n_splits):
            run_name = self.get_run_name(i_fold)
            model = self.model_cls(run_name, self.params, self.fit_params)
            model.load_model(self.model_dir)
            pred = model.predict(x_test)
            preds.append(pred)

        preds = np.mean(preds, axis=0)
        self.save_submission(preds)

        logger.info("🎉 Finish Prediction and Create submission.csv!!")

    @property
    def model_dir(self) -> Path:
        """学習済みモデルの保存先ディレクトリを取得

        Returns:
            str: 学習済みモデルの保存先ディレクトリ
        """
        return Path(self.config.models_dir, self.exp_number, self.date_str)

    @property
    def save_figs_dir(self) -> Path:
        """学習済みモデルの保存先ディレクトリを取得

        Returns:
            str: 学習済みモデルの保存先ディレクトリ
        """
        return Path(self.config.figs_dir, self.exp_number, self.date_str)

    def load_x_train(self) -> pd.DataFrame:
        """学習用データの説明変数の読み込み

        Returns:
            pd.DataFrame: 学習用データ説明変数
        """
        dfs = [
            pd.read_pickle(Path(self.config.features_dir, self.exp_number, f"{feat}_train.pkl"))
            for feat in self.config.features
        ]
        return pd.concat(dfs, axis=1)

    def load_y_train(self) -> pd.DataFrame:
        """学習用データの目的変数の読み込み

        Returns:
            pd.DataFrame: 学習用データ目的変数
        """
        return pd.read_pickle(Path(self.config.features_dir, self.exp_number, self.config.target_train_file))

    def load_x_test(self) -> pd.DataFrame:
        """テスト用データの説明変数の読み込み

        Returns:
            pd.DataFrame: 学習用データ説明変数
        """
        dfs = [
            pd.read_pickle(Path(self.config.features_dir, self.exp_number, f"{feat}_test.pkl"))
            for feat in self.config.features
        ]
        return pd.concat(dfs, axis=1)

    def load_group(self) -> Optional[pd.DataFrame]:
        """GroupKFoldで使用するグループの読み込み

        Returns:
            pd.DataFrame: グループのpd.DataFrame
        """
        if self.config.group_col and self.config.group_train_file:
            return pd.read_pickle(Path(self.config.features_dir, self.exp_number, self.config.group_train_file))
        else:
            return None

    def get_run_name(self, i_fold: int) -> str:
        """実験名を取得

        Returns:
            str: 実験名
        """
        return f"{self.config.model_name}_fold{i_fold}"

    def load_fold_model(self, i_fold: int) -> Any:
        """学習済みモデルを読み込む

        Args:
            i_fold (int): fold番号

        Returns:
            Any: 学習済みモデル
        """
        run_name = self.get_run_name(i_fold)
        return self.model_cls(run_name, self.params, self.fit_params)

    def save_oof(self, oof: np.ndarray) -> None:
        """oofを保存

        Args:
            oof (np.ndarray): oof
        """
        oof_df = pd.DataFrame(oof, columns=["oof"])
        save_dir = Path(self.config.oof_dir, self.exp_number)
        save_dir.mkdir(parents=True, exist_ok=True)
        oof_df.to_csv(Path(save_dir, f"{self.date_str}.csv"), index=False)

    def save_fig(self, fig) -> None:
        """oofを保存

        Args:
            oof (np.ndarray): oof
        """
        save_dir = Path(self.config.figs_dir, self.exp_number)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir, f"{self.date_str}_feature_importance.png"))

    def save_submission(self, preds: List[Any]) -> None:
        """推論結果を保存

        Args:
            preds (np.ndarray): テストデータの予測値
        """
        cliped_preds = np.clip(np.array(preds), 1, 10)
        # sub_df = pd.read_csv(self.config.submission_csv_path)
        # sub_df.iloc[:, 0] = cliped_preds

        save_dir = Path(self.config.submission_dir, self.exp_number)
        save_dir.mkdir(parents=True, exist_ok=True)
        # sub_df.to_csv(Path(save_dir, f"{self.date_str}_submission.csv"), index=False)
        np.save(Path(self.config.submission_dir, self.exp_number, f"{self.date_str}_submission.npy"), cliped_preds)


if __name__ == "__main__":

    # このファイルがあるディレクトリの絶対パスを取得 -> 実験番号(ex. exp001)及びconfig.yamlの絶対パスを取得
    abs_path: str = os.path.dirname(os.path.abspath(__file__))
    exp_number: str = abs_path.split("/")[-1]
    config: Union[DictConfig, ListConfig] = OmegaConf.load(Path(abs_path, "config.yaml"))

    date_str: str = datetime.now(timezone("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")

    logger.add(Path(config.logs_dir, exp_number, f"{date_str}.log"))

    runner = Runner(config, exp_number, date_str)
    runner.train_cv()
    runner.predict_cv()
