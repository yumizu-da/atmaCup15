from typing import Any, Tuple, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.models import LightGBM


def select_model(config: Union[DictConfig, ListConfig]) -> Tuple[Any, Any, Any]:
    """モデルを選択する

    なおparams/fit_paramsに関しては, to_container関数を用いて辞書型に変換する
    辞書型に変換しないと, 学習時にエラーが発生するため

    Args:
        config (Union[DictConfig, ListConfig]): configファイル.

    Raises:
        ValueError: あらかじめ定義されているモデル以外が指定された場合

    Returns:
        Tuple[Any, dict, dict]: モデル, 学習時のパラメータ, 検証時のパラメータ
    """
    if config.model_name == "LightGBM":
        mode_cls = LightGBM
        params = OmegaConf.to_container(config.lgb_params)
        fit_params = OmegaConf.to_container(config.lgb_fit_params)
    else:
        raise ValueError(f"Unknown model: {config.model_name}")

    return mode_cls, params, fit_params
