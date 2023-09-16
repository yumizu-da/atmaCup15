# atmacup #15

atmacup #15のコンペ終了後に, Discussionを参考にLateSUbしたリポジトリです.

<https://www.guruguru.science/competitions/21/>

- 実際のコンペ終了時の順位/スコア

  - Public: 1.1888 20位
  - Private: 1.1557 24位

- 本リポジトリでのスコア

  - Public: 1.2014 62位相当
  - Private: 1.1557 37位相当

## Environment

### dockerコンテナ ビルド & 起動

```bash
docker compose up -d --build dev
```

### Run

#### 特徴量作成

  ```bash
  python3 src/exp/exp001_seen/feature_engineering.py
  python3 src/exp/exp001_unseen/feature_engineering.py
  ```

#### 学習/推論

  ```bash
  python3 src/exp/exp001_seen/run.py
  python3 src/exp/exp001_unseen/run.py
  ```

#### 提出csv作成

  ```bash
  python3 src/exp/exp001_concat/concat.py
  ```
# atmacup15
