# 家計調査ダッシュボード — 家計の立ち位置診断と貯蓄分析

総務省の公的統計（家計調査・全国家計構造調査・一般用ミクロデータ）を使って、
「あなたの家計は世の中と比べてどうか」を診断する Streamlit アプリと、
「貯蓄余力の高い世帯は何が違うか」の機械学習分析です。

チーム開発課題（AIProgrammingⅣ）の成果物。

## 何ができるか

- 🩺 **あなたの家計診断** — 貯蓄のパーセンタイル（分布の下から◯%地点）、同年代×同収入・同収入階級・同年代との比較、黒字率診断、家計タイプマップ（参考）
- 💡 **統計にみる貯蓄の傾向** — 擬似個票4.6万世帯の機械学習分析（ランダムフォレスト ROC-AUC 0.774）と、黒字率・資産構成の集計分析
- 📈 **データとトレンド** — 24年分の時系列、年代で絞った収入×貯蓄の関係

分析の設計判断（何を・なぜ・どう判断したか）は [`kansei/DESIGN_DECISIONS.md`](kansei/DESIGN_DECISIONS.md) に記録しています。

## セットアップ

必要なもの: Python 3.12+（開発は3.14で実施）

```bash
git clone https://github.com/kyiku/household-finance-analysis.git
cd household-finance-analysis

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r kansei/requirements.txt
```

## 起動方法

分析済みデータ（`kansei/data/*.csv`）は同梱しているので、**そのまま起動できます**:

```bash
cd kansei
streamlit run app.py
```

ブラウザで http://localhost:8501 が開きます。

## テスト

```bash
cd kansei
pytest tests -q                    # 93テスト
pytest tests --cov=src             # カバレッジ(src/ 96%+)
```

## データの再取得（任意）

### e-Stat API（家計調査・全国家計構造調査）

1. https://www.e-stat.go.jp/mypage/user/preregister でユーザー登録し、マイページからアプリケーションIDを発行
2. リポジトリ直下に `.env` を作成: `ESTAT_APP_ID=あなたのID`（**`.env` はコミット禁止**。`.gitignore` 済み）
3. 取得スクリプトを実行:

```bash
cd kansei
python scripts/fetch_benchmark_data.py   # 貯蓄分布・年齢×収入クロス表 → data/
```

家計調査の基本5データセット（収入階級別・年齢階級別など）の取得手順は
`notebooks/kakei_data_fetch.ipynb` を参照してください。

### 一般用ミクロデータ（機械学習の学習データ）

利用規約に基づき **各自でダウンロード** してください（リポジトリには含めていません）:

1. https://www.nstac.go.jp/use/archives/ippan-microdata/request/ で利用規約に同意し、利用者情報を登録
2. 「平成21年全国消費実態調査（十大費目）」`ippan_2009zensho.zip` をダウンロード
3. `kansei/microdata/` に展開（`kansei/microdata/ippan_2009zensho/ippan_2009zensho_z_dataset.csv` になる配置）
4. モデルを学習:

```bash
cd kansei
python scripts/train_saver_model.py      # 結果CSVを data/ に出力(アプリが表示)
```

※ 学習済みの結果CSV（`kansei/data/microdata_*.csv`）は同梱しているため、
モデルを再学習しない限りこの手順は不要です。

## プロジェクト構成

```
kansei/
  app.py                 # Streamlit エントリポイント(3タブ)
  src/                   # 純粋ロジック(すべてユニットテスト済み)
    periods.py           #   e-Stat時期コードのパース
    data_loader.py       #   CSV読み込み・整形
    benchmark.py         #   直近1年ローリング平均ベンチマーク
    percentile.py        #   貯蓄分布のパーセンタイル推定(線形補間)
    cross_benchmark.py   #   同年代×同収入ベンチマーク
    clustering.py        #   家計タイプマップ(KMeans・参考情報)
    analysis.py          #   黒字率・貯蓄内訳などの集計
    microdata.py         #   擬似ミクロデータの特徴量構築
  ui/                    # Streamlit 画面(タブごと)
  scripts/               # データ取得・モデル学習(事前実行)
  tests/                 # pytest(93件) + AppTestスモーク
  data/                  # 取得済みCSV + モデル結果
  DESIGN_DECISIONS.md    # 意思決定の記録(D1〜D11)
notebooks/               # データ取得・探索ノートブック(Colab版含む)
```

## データ出典・利用上の注意

- 出典: 総務省統計局「家計調査」「2019年全国家計構造調査」（e-Stat API 経由で取得・加工）
- 「一般用ミクロデータ（平成21年全国消費実態調査）」（総務省統計局）を加工して作成
- 一般用ミクロデータは擬似データであり、**分析結果は実証研究の結果とみなせません**（教育・演習用）
- 「二人以上世帯」と「勤労者世帯」など対象の異なる統計が混在するため、タブ間で水準を直接比較しないでください
