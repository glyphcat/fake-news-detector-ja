import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    FULL_FAKE_DATA_NAME,
    PARTIAL_FAKE_DATA_NAME,
    PROCESSED_DIR,
    RANDOM_STATE,
    REAL_FAKE_DATA_NAME,
    SPLIT_FINETUNE_DIR,
    SPLIT_PRETUNE_DIR,
    TEST_DATA_NAME,
    TEST_SIZE,
    TRAIN_DATA_NAME,
    VALIDATION_DATA_NAME,
    VALIDATION_SIZE,
)
from utils import Utils


def split_data(combined_df: pd.DataFrame, dir_path: str):
    # 学習・検証データ、テストデータに分割
    train_val_df, test_df = train_test_split(
        combined_df,
        test_size=TEST_SIZE,
        stratify=combined_df["is_fake"],
        random_state=RANDOM_STATE,
    )

    # 学習データ、検証データに分割
    train_df, validation_df = train_test_split(
        train_val_df,
        test_size=VALIDATION_SIZE,
        stratify=train_val_df["is_fake"],
        random_state=RANDOM_STATE,
    )
    Utils.export_df_as_csv(test_df, dir_path, TEST_DATA_NAME)
    Utils.export_df_as_csv(train_df, dir_path, TRAIN_DATA_NAME)
    Utils.export_df_as_csv(validation_df, dir_path, VALIDATION_DATA_NAME)


if __name__ == "__main__":
    full_fake_df = pd.read_csv(os.path.join(PROCESSED_DIR, FULL_FAKE_DATA_NAME))
    partial_fake_df = pd.read_csv(os.path.join(PROCESSED_DIR, PARTIAL_FAKE_DATA_NAME))
    real_df = pd.read_csv(os.path.join(PROCESSED_DIR, REAL_FAKE_DATA_NAME))

    # ラベルのデータ数を同数にする
    min_size = min(len(full_fake_df), len(real_df))
    balanced_full_fake_df = full_fake_df.sample(min_size, random_state=RANDOM_STATE)
    balanced_real_df = real_df.sample(min_size, random_state=RANDOM_STATE)

    # 最初の学習: 完全fakeと完全realのみ
    pretrain_combined_df = pd.concat([balanced_full_fake_df, balanced_real_df]).sample(
        frac=1, random_state=RANDOM_STATE
    )
    split_data(pretrain_combined_df, SPLIT_PRETUNE_DIR)

    # 本学習: 一部fakeの曖昧データも含んで学習
    finetune_combined_df = pd.concat(
        [balanced_full_fake_df, partial_fake_df, balanced_real_df]
    ).sample(frac=1, random_state=RANDOM_STATE)
    split_data(pretrain_combined_df, SPLIT_FINETUNE_DIR)
