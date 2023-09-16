from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMModel, early_stopping, log_evaluation

from src.models.base import BaseModel


class LightGBM(BaseModel):
    """LightGBM model class

    sckit-learn APIを使用してLightGBMモデルを構築
    評価関数等を独自で定義する場合は、Python APIを使用する必要がある

    Attributes:
        name (str): モデル名
        params (Dict[str, Any]): モデルのパラメータ
        fit_params (Dict[str, Any]): モデルの学習時のパラメータ

    Examples:
        >>> from src.models.model_lgbm import LightGBM
        >>> model = LightGBM(name="lgbm", params=lgb_params, fit_params=lgb_fit_params)
        >>> model.fit(X_train, y_train, X_valid, y_valid)
        >>> preds = model.predict(X_test)
    """

    def __init__(self, name: str, params: Dict[str, Any], fit_params: Dict[str, Any]):
        super().__init__(name, params, fit_params)

        self.stopping_rounds = self.fit_params.pop("early_stopping_rounds")
        self.verbose = self.fit_params.pop("verbose")
        self.log_evaluation = self.fit_params.pop("log_evaluation")
        self.feature_importance_list: List[pd.DataFrame] = []

    def build_model(self, **kwargs: Any) -> LGBMModel:
        """LightGBMモデルを構築する

        Returns:
            model (LGBMModel): LightGBMモデル
        """
        return LGBMModel(**self.params)

    def fit(self, X_train: Any, y_train: Any, X_valid: Optional[Any] = None, y_valid: Optional[Any] = None) -> None:
        """モデルを学習する

        Args:
            X_train (Any): 学習データの説明変数
            y_train (Any): 学習データの目的変数
            X_valid (Optional[Any]): 検証データの説明変数
            y_valid (Optional[Any]): 検証データの目的変数
        """
        self.model = self.build_model()
        if X_valid is not None and y_valid is not None:
            self.fit_params["eval_set"] = [(X_valid, y_valid)]
            self.fit_params["callbacks"] = [
                early_stopping(stopping_rounds=self.stopping_rounds, verbose=self.verbose),
                log_evaluation(self.log_evaluation),
            ]
        self.model.fit(X_train, y_train, **self.fit_params)
        f_importance = self.feature_importance(X_train)
        self.feature_importance_list.append(f_importance)

    def predict(self, X: Any) -> np.ndarray:
        """モデルを使って予測を行う

        Args:
            X (Any): 予測データの説明変数

        Returns:
            preds (np.ndarray): 予測結果
        """
        return self.model.predict(X)

    def feature_importance(self, tr_x: Any) -> pd.DataFrame:
        """モデルの特徴量の重要度を返す

        Returns:
            feature_importance (np.ndarray): 特徴量の重要度
        """
        f_importance = self.model.feature_importances_
        f_importance = f_importance / np.sum(f_importance)
        return pd.DataFrame({"feature": tr_x.columns, "feature_importance": f_importance})
