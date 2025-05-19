# fake-news-detector-ja

## 概要

日本語ニュース記事についてフェイクニュースを判別するモデル。

日本語BERTモデルをファインチューニングし、学習はデータの曖昧さに応じて２段階に分けて実施し、精度の向上を図っている。

## 使用モデル

- [東北大BERT日本語pretrained](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3)

## 使用データセット

- 実際のニュース: [日本語Wikiニュースデータセット]([hpprc/jawiki-news](https://huggingface.co/datasets/hpprc/jawiki-news/tree/main))
- フェイクニュース: [日本語フェイクニュースデータセット](https://huggingface.co/datasets/p1atdev/fake-news-jp)

\* 実際のニュースデータには比較的新しい2024年度の記事が含まれる

\* フェイクニュースデータセットはGithubで公開されている[日本語フェイクニュースデータセット](https://github.com/tanreinama/Japanese-Fakenews-Dataset)をhugging face向けに変換したもの

## 技術スタック/使用ツール

- python: 3.12.8
- uv: バージョン管理・仮想環境
- pylance, black, isort: linter, formatter
- pytest: テスト
- hugging face: モデル・データセットの使用

## ディレクトリ構造
```
.
├── config.py
├── LICENSE
├── pyproject.toml
├── README.md
├── src
│   ├── evaluate.py
│   ├── prepare_data.py
│   ├── split_data.py
│   ├── train.py
│   └── utils.py
├── test
│   ├── helpers
│   │   └── script_runner.py
│   ├── test_finetune_evaluate.py
│   ├── test_finetune_train.py
│   ├── test_pipeline.py
│   ├── test_pretune_evaluate.py
│   └── test_pretune_train.py
└── uv.lock
```

## 処理の流れ

学習、評価は次の二段階に分けて行う。

- 段階１: 完全にフェイク/完全にリアルなデータのみ用いて学習
- 段階２：完全にフェイク/完全にリアルなデータに加え、一部のみフェイクのニュース本文を用いて学習

\* 便宜上、第一段階をPretune, 第二段階をFinetuneと表現する

### 1. データ前処理

各データセットより、ニュース本文のみ抜き出す。
本文中の改行コードなどは削除する。
それぞれの属性に応じて学習用のラベル「is_fake」を数値で付与する。

|列名|概要|
|---|------------|
|id |各記事の識別子 |
|text|ニュース本文 |
|is_fake|リアル/フェイクのラベル(0: リアル、1: フェイク) |

またフェイクニュース記事には次の２種類がある。

- 記事全体がフェイク
- 記事の一部分がフェイク

これらはどちらもフェイクニュースと判定させることとし、is_fake = １とラベリングする。

(今後３つのタイプについて判定させたい場合は、ラベルを0~2の3値に変更する)

最終的に次のようなCSV形式のデータセットが出力される。

|名称|概要|
|---|------------|
|full_fake |完全にフェイクのニュース記事 |
|partial_fake|一部フェイクのニュース記事 |
|real|実際のニュース記事 |

### 2. 学習用データセット作成

#### 第一段階(Pretune)学習・評価用データセット

第一弾の学習Pretune段階では、確度の高いデータを使用してチューニングする。

そのため、full_fakeデータとrealデータのみを使用してデータセットを作成する。

#### 第二段階（Finetune）学習・評価用データセット

第二弾の学習Finetune段階では、信頼性の低い曖昧なデータも使用してチューニングする。

したがってpartial_fakeも含めた全種類のデータを使用してデータセットを作成する。

#### データの分割

Pretune, Finetuneともに、次の3つのデータセットを用意する。

|名称|概要|
|---|------------|
|train |学習用データ |
|validation|学習中の検証用データ |
|test|学習後のモデル評価のためのデータ |

(train, validation)とtestの比率は**8:2**、train:validationの比率は**7:3**として分割する。

また、full_fakeデータとrealデータは同数になるように調整する。

### 3. 学習

学習は次の２段階で行われる。

|学習段階|概要|
|---|------------|
|Phase1: Pretune |信頼性の高い、realおよびfull_fakeのみの混合データで行う |
|Phase2: Finetune|一部のみフェイクのpartial_fakeを含んだ混合データで行う |


Phase1では日本語BERTモデルをベースに学習を行い、

Phase2ではPhase1で学習したモデルをベースに学習を行う。

### 4. 評価

学習段階ごとのモデルの評価はpartial_fakeを含んだtestデータで行う。

## セットアップ/使用方法

### 環境構築

clone後、次のコマンドで依存関係をインストールする。

```bash
uv sync
```

### 仮想環境の起動/終了

- 起動
```bash
. .venv/bin/activate
```

- 終了

```bash
deactivate
```

### パイプライン実行

次を実行すると、データ前処理からモデル評価までの一連の処理がパイプラインで実行される。

学習は自動で第二弾学習（Finetune）まで実行される。

```bash
pytest -s test/test_pipeline.py
```

### 各処理の個別実行

#### 1. データ前処理

次を実行するとdata/processed/に各種処理ずみデータセットがCSV形式で出力される。

動作確認用にデータ件数を抑えるには--testオプションを使用する。

```bash
python python src/prepare_data.py
python python src/prepare_data.py --test
```

#### 2. データセット作成

次を実行するとdata/split/に分割ずみデータセットがCSV形式で出力される。

データ件数はsrc/processed/内の各ファイルのデータ数に依存する。

そのため動作確認用にデータ件数を制限したい場合は、データ前処理時に--testオプションを使用する。

```bash
python python src/split_data.py
```

#### 3. 学習

次を実行するとdata/split/のデータセットを使用してファインチューニングが実行される。

各学習段階は--phaseオプションで指定する。

```bash
python src/train.py --phase [option: pretune or finetune]
```

学習後のモデルデータおよびログファイルはmodel/に出力される。

#### 4. 評価

次を実行するとmodel/内のモデルデータに対して評価が実行される。

評価対象のモデルは--phaseオプションで指定する。

```bash
python src/evaluate.py --phase [option: pretune or finetune]
```

### パラメータの変更

現在パラメータの一部はconfig.pyで管理している。

必要に応じてエポック数など上書きして調整する。

### 評価結果

モデルの評価結果は次にJSON形式で出力される。

```
model/{phase}/evaluate/results.json
```

## TODO



