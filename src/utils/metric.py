import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score


def metric(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """評価指標を選択して計算する

    Args:
        method (str): 評価指標
        y_true (np.ndarray): 正解ラベル
        y_pred (np.ndarray): 予測ラベル

    Raises:
        ValueError: あらかじめ定義されている評価指標以外が指定された場合

    Returns:
        float: 評価指標の値
    """
    if method == "MAE":
        return mean_absolute_error(y_true, y_pred)
    elif method == "RMSE":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif method == "AUC":
        return roc_auc_score(y_true, y_pred)
    elif method == "logloss":
        return log_loss(y_true, y_pred, eps=1e-15, normalize=True)
    elif method == "Accuracy":
        return accuracy_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown method: {method}")
