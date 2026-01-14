from dataclasses import dataclass


@dataclass(frozen=True)
class Consts:
    """
    TODO: 環境変数から各定数を組み立てるよう変更
    1. load_dotenv()をエントリポイントで実行
    2. 環境変数を読み込みConsts内で各定数の組み立て
    3. constsは一度のみインスタンス化する -> Singletonに変更
    """

    PROJECT_NAME = "fake-news-detector-ja"

    # === Datasets ===
    FAKE_NEWS_DATASET = "p1atdev/fake-news-jp"
    # see https://huggingface.co/datasets/p1atdev/fake-news-jp

    REAL_NEWS_DATASET = "hpprc/jawiki-news"
    # see https://huggingface.co/datasets/hpprc/jawiki-news

    # === Model ===
    BASE_MODEL = "tohoku-nlp/bert-base-japanese-v3"
    # see https://huggingface.co/tohoku-nlp/bert-base-japanese-v3

    # === Path ===
    # google driveプロジェクトルート
    GOOGLE_DRIVE_ROOT = f"/content/drive/MyDrive/{PROJECT_NAME}"

    DATA_DIR = "data"
    PROCESSED_DIR = f"{DATA_DIR}/processed"
    SPLIT_DIR = f"{DATA_DIR}/split"

    # 最初の学習データ
    SPLIT_PRETUNE_DIR = f"{DATA_DIR}/split/pretune"
    # 全データを使用した本学習データ
    SPLIT_FINETUNE_DIR = f"{DATA_DIR}/split/finetune"

    # model出力
    MODEL_DIR = "model"
    MODEL_LOG_DIR = "logs"
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
