from sklearn.model_selection import BaseCrossValidator, GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold


def select_cv_method(method: str, n_splits: int, shuffle: bool = True, random_state: int = 42) -> BaseCrossValidator:
    """Validationの方法を選択する

    Args:
        method (str): validation手法
        n_splits (int): Fold数
        shuffle (bool, optional): shuffleの有無. Defaults to True.
        random_state (int, optional): seed値. Defaults to 42.

    Raises:
        ValueError: あらかじめ定義されている手法以外が指定された場合

    Returns:
        BaseCrossValidator: validation手法
    """
    if method == "KFold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    elif method == "StratifiedKFold":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    elif method == "GroupKFold":
        return GroupKFold(n_splits=n_splits)
    elif method == "StratifiedGroupKFold":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")
