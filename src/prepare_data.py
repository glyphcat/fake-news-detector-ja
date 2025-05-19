from datasets import load_dataset

from config import (
    FAKE_NEWS_DATASET,
    FULL_FAKE_DATA_NAME,
    PARTIAL_FAKE_DATA_NAME,
    PROCESSED_DIR,
    REAL_DATA_NAME,
    REAL_NEWS_DATASET,
    SAMPLE_NUM_FOR_TEST,
)
from utils import Utils


# load fake news
def get_processed_fake():
    fake_ds = load_dataset(FAKE_NEWS_DATASET, split="train", trust_remote_code=True)
    # TODO: 型チェック対応
    fake_df = fake_ds.to_pandas().copy()
    # partial_fake_df = fake_df[fake_df["fake_type"].isin(["partial_gpt2"])].drop(
    #     ["fake_type"], axis=1
    # ).copy()
    # fake_df = (
    #     fake_df.drop(["nchar_real", "nchar_fake", "fake_type"], axis=1)
    #     .rename(columns={"context": "text"})
    # )
    fake_df = fake_df[["id", "context", "fake_type"]].rename(
        columns={"context": "text"}
    )

    # テキストの全てがfake
    full_fake_df = fake_df[fake_df["fake_type"] == "full_gpt2"].copy()
    full_fake_df["is_fake"] = 1

    # テキストの一部がfake
    partial_fake_df = fake_df[fake_df["fake_type"] == "partial_gpt2"].copy()
    partial_fake_df["is_fake"] = 1
    return full_fake_df, partial_fake_df


# load real news
def get_processed_real():
    real_ds = load_dataset(REAL_NEWS_DATASET, split="train")
    real_df = real_ds.to_pandas().copy()
    # real_df = real_df.drop(
    #     [
    #         "title",
    #         "paragraphs",
    #         "abstract",
    #         "wikitext",
    #         "date_created",
    #         "date_modified",
    #         "templates",
    #         "url",
    #     ],
    #     axis=1,
    # ).copy()
    real_df = real_df[["id", "text"]]
    # 改行コードと日付【YYYY年MM月DD日】の削除
    real_df["text"] = (
        real_df["text"]
        .str.replace(r"[\r\n]+", "", regex=True)
        .str.replace(r"【\d{4}年\d{1,2}月\d{1,2}日】", "", regex=True)
        .str.strip()
    )
    real_df["is_fake"] = 0
    return real_df


if __name__ == "__main__":
    args = Utils.parse_test_args()
    full_fake_df, partial_fake_df = get_processed_fake()
    real_df = get_processed_real()

    # テスト時はデータ数制限
    if args.is_test_run:
        full_fake_df = full_fake_df.head(SAMPLE_NUM_FOR_TEST)
        partial_fake_df = partial_fake_df.head(SAMPLE_NUM_FOR_TEST)
        real_df = real_df.head(SAMPLE_NUM_FOR_TEST)

    # 処理ずみデータの書き込み
    Utils.export_df_as_csv(full_fake_df, PROCESSED_DIR, FULL_FAKE_DATA_NAME)
    Utils.export_df_as_csv(partial_fake_df, PROCESSED_DIR, PARTIAL_FAKE_DATA_NAME)
    Utils.export_df_as_csv(real_df, PROCESSED_DIR, REAL_DATA_NAME)
