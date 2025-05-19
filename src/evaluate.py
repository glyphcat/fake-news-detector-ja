import json
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from config import MODEL_DIR, SPLIT_DIR
from utils import TrainPhase, Utils

trusted_only_model_path = os.path.join(MODEL_DIR, TrainPhase.PRETUNE)
full_model_path = os.path.join(MODEL_DIR, TrainPhase.FINETUNE)

tokenizer = None
model = None
# バッチごとの評価値
all_preds = []
all_labels = []


def load_tokenizer_and_model(train_phase: TrainPhase):
    trained_model_path = os.path.join(MODEL_DIR, train_phase)
    if not Path(trained_model_path).exists():
        print(f"Error: モデルディレクトリが存在しません: {trained_model_path}")
        sys.exit(1)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
    global model
    model = AutoModelForSequenceClassification.from_pretrained(trained_model_path)


def tokenize(batch):
    tokenized = tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=512
    )
    tokenized["labels"] = batch["is_fake"]
    return tokenized


def get_tokenized_dataset(train_phase: TrainPhase):
    # NOTE: 精度の改善を評価するため、モデルの評価はtrain_phaseに関係なく、全データを使用するように変更
    test_df = pd.read_csv(f"{SPLIT_DIR}/{TrainPhase.FINETUNE}/test.csv")
    # test_df = pd.read_csv(f"{SPLIT_DIR}/{train_phase}/test.csv")
    test_ds = Dataset.from_pandas(test_df)
    tokenized_test = test_ds.map(tokenize, batched=True)
    return tokenized_test


def compute_metrics(pred=None, compute_result=False):
    # === batch_eval_metricsに対応する場合 ===
    # global all_preds, all_labels
    # if compute_result:
    #     precision, recall, f1, _ = precision_recall_fscore_support(
    #         all_labels, all_preds, average="binary"
    #     )
    #     acc = accuracy_score(all_labels, all_preds)
    #     all_preds = []
    #     all_labels = []
    #     return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    # else:
    #     labels = pred.label_ids
    #     preds = pred.predictions.argmax(-1)
    #     all_preds.extend(preds.tolist())
    #     all_labels.extend(labels.tolist())
    #     return {}
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def evaluate(train_phase: TrainPhase):
    test_ds = get_tokenized_dataset(train_phase)
    output_dir = Utils.get_model_eval_output_path(train_phase)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",
        per_device_eval_batch_size=8,
        metric_for_best_model="accuracy",
        batch_eval_metrics=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        train_dataset=None,
    )

    result = trainer.evaluate(test_ds)
    print(
        f"📝 {train_phase} 学習モデル評価結果\n",
        result,
    )
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = Utils.parse_train_phase_args()
    train_phase = args.phase
    load_tokenizer_and_model(train_phase)
    evaluate(train_phase)
