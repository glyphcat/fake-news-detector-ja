# === Datasets ===

# see https://huggingface.co/datasets/p1atdev/fake-news-jp
FAKE_NEWS_DATASET = "p1atdev/fake-news-jp"
# see https://huggingface.co/datasets/hpprc/jawiki-news
REAL_NEWS_DATASET = "hpprc/jawiki-news"

# === Model ===
# see https://huggingface.co/tohoku-nlp/bert-base-japanese-v3
BASE_MODEL = "tohoku-nlp/bert-base-japanese-v3"


#  === Path ===
DATA_DIR = "data"
PROCESSED_DIR = f"{DATA_DIR}/processed"
SPLIT_DIR = f"{DATA_DIR}/split"

# 最初の学習データ
SPLIT_PRETUNE_DIR = f"{DATA_DIR}/split/pretune"
# 全データを使用した本学習データ
SPLIT_FINETUNE_DIR = f"{DATA_DIR}/split/finetune"

# model出力
MODEL_DIR = "model"
# 学習ログ
MODEL_LOG_DIR = "logs"
# 評価結果
MODEL_EVAL_DIR = "evaluate"

# データファイル
FULL_FAKE_DATA_NAME = "full_fake.csv"
PARTIAL_FAKE_DATA_NAME = "partial_fake.csv"
REAL_FAKE_DATA_NAME = "real_fake.csv"

TRAIN_DATA_NAME = "train.csv"
VALIDATION_DATA_NAME = "validation.csv"
TEST_DATA_NAME = "test.csv"


#  === Parameter ===

# for data
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.3
RANDOM_STATE = 42

# for train
TRAIN_EPOCHS = 1
