import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class BaseModel(metaclass=ABCMeta):
    """モデル作成の基底クラス

    Attributes:
        name (str): モデル名
        params (Dict[str, Any]): モデルのパラメータ
        fit_params (Dict[str, Any]): モデルの学習時のパラメータ

    Raises:
        NotImplementedError: 子クラスでbuild_modelメソッドを実装していない場合に発生
    """

    def __init__(self, name: str, params: Dict[str, Any], fit_params: Dict[str, Any]):
        self.name = name
        self.params = params.copy()
        self.fit_params = fit_params.copy()

    @abstractmethod
    def build_model(self, **kwargs: Any) -> Any:
        """モデルを構築する抽象メソッド

        Raises:
            NotImplementedError: 子クラスでこのメソッドが実装されていない場合
        """
        raise NotImplementedError("build_modelメソッドが実装されていません")

    def fit(self, X_train: Any, y_train: Any, X_valid: Optional[Any] = None, y_valid: Optional[Any] = None) -> None:
        """モデルを学習する

        Args:
            X_train (Any): 学習データの説明変数
            y_train (Any): 学習データの目的変数
            X_valid (Optional[Any]): 検証データの説明変数（デフォルト値: None）
            y_valid (Optional[Any]): 検証データの目的変数（デフォルト値: None）
        """
        self.model = self.build_model()
        self.model.fit(X_train, y_train)

    def predict(self, X: Any) -> Any:
        """モデルを使って予測を行う

        Args:
            X (Any): 予測データの説明変数

        Returns:
            preds (Any): 予測結果
        """
        return self.model.predict(X)

    def save_model(self, save_dir: Path) -> None:
        """モデルを保存する

        Args:
            save_dir (Path): 保存先のディレクトリ
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, f"{self.name}.pkl")
        with open(save_path, mode="wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, save_dir: Path) -> None:
        """モデルを読み込む

        Args:
            save_dir (Path): 保存先のディレクトリ
        """
        save_path = Path(save_dir, f"{self.name}.pkl")
        with open(save_path, mode="rb") as f:
            self.model = pickle.load(f)
