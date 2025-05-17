import argparse
import os
from enum import StrEnum
from pathlib import Path

from pandas import DataFrame

from config import MODEL_DIR, MODEL_EVAL_DIR, MODEL_LOG_DIR

# from typing import Literal


# 学習ステップ
# TrainType = Literal["trusted", "full"]


class TrainPhase(StrEnum):
    PRETUNE = "pretune"
    FINETUNE = "finetune"

    @classmethod
    def get_names(cls):
        return [item.name for item in cls]

    @classmethod
    def get_values(cls):
        return [item.value for item in cls]


class EnvPhase(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"

    @classmethod
    def get_names(cls):
        return [item.name for item in cls]

    @classmethod
    def get_values(cls):
        return [item.value for item in cls]


class Utils:
    @staticmethod
    def export_df_as_csv(df: DataFrame, dir_path: str, file_name: str):
        file_path = os.path.join(dir_path, file_name)
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)

    @staticmethod
    def get_model_train_output_path(train_phase: str):
        output_dir = os.path.join(MODEL_DIR, train_phase)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logging_dir = os.path.join(output_dir, MODEL_LOG_DIR)
        Path(logging_dir).mkdir(parents=True, exist_ok=True)
        return output_dir, logging_dir

    @staticmethod
    def get_model_eval_output_path(train_phase: str):
        output_dir, _ = Utils.get_model_train_output_path(train_phase)
        output_dir = os.path.join(output_dir, MODEL_EVAL_DIR)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def parse_env_phase_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--env",
            type=str,
            choices=EnvPhase.get_values(),
            required=True,
            help="Use 'cloud' for training at Google Colab, 'local' for test training at local",
        )

    @staticmethod
    def parse_train_phase_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--phase",
            type=str,
            choices=TrainPhase.get_values(),
            required=True,
            help="Use 'pretune' for training by trusted data only, 'finetune' for training by full data",
        )
        return parser.parse_args()

    @staticmethod
    def parse_test_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--test", action="store_true", dest="is_test_run", required=False
        )
        return parser.parse_args()
