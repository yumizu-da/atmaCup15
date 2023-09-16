import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

if __name__ == "__main__":

    abs_path: str = os.path.dirname(os.path.abspath(__file__))
    exp_number: str = abs_path.split("/")[-1].split("_")[0]
    config_path: Path = Path(abs_path, "config.yaml")

    unseen_sub_path = sorted(Path("results/submission", f"{exp_number}_unseen").glob("*.npy"))[-1]
    seen_sub_path = sorted(Path("results/submission", f"{exp_number}_seen").glob("*.npy"))[-1]

    exp_unseen: np.ndarray = np.load(unseen_sub_path)
    exp_seen: np.ndarray = np.load(seen_sub_path)

    config: Union[DictConfig, ListConfig] = OmegaConf.load(config_path)
    train_df = pd.read_csv(config.train_csv_path)
    test_df = pd.read_csv(config.test_csv_path)
    anime_df = pd.read_csv(config.anime_csv_path)

    test_unseen_df = test_df[~test_df["user_id"].isin(train_df["user_id"])]
    test_seen_df = test_df[test_df["user_id"].isin(train_df["user_id"])]

    test_unseen_df["pred"] = exp_unseen
    test_seen_df["pred"] = exp_seen

    pred = pd.concat([test_unseen_df["pred"], test_seen_df["pred"]]).sort_index()
    sub_df = pd.read_csv(config.submission_csv_path)
    sub_df["score"] = pred

    os.makedirs(f"results/submission/{exp_number}", exist_ok=True)
    sub_df.to_csv(f"results/submission/{exp_number}/{exp_number}_submission.csv", index=False)
