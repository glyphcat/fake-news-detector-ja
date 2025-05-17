import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from config import BASE_MODEL, MODEL_DIR, SPLIT_DIR, TRAIN_EPOCHS
from utils import TrainPhase, Utils

tokenizer = None
model = None


def load_tokenizer_and_model(train_phase: TrainPhase):
    pretune_model_path, _ = Utils.get_model_train_output_path(TrainPhase.PRETUNE)
    model_path_or_name = (
        BASE_MODEL if train_phase == TrainPhase.PRETUNE else pretune_model_path
    )
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    global model
    model = AutoModelForSequenceClassification.from_pretrained(model_path_or_name)


def get_tokenized_dataset(train_phase: TrainPhase):
    train_df = pd.read_csv(f"{SPLIT_DIR}/{train_phase}/train.csv")
    validation_df = pd.read_csv(f"{SPLIT_DIR}/{train_phase}/validation.csv")
    train_ds = Dataset.from_pandas(train_df)
    validation_ds = Dataset.from_pandas(validation_df)
    tokenized_train = train_ds.map(tokenize, batched=True)
    tokenized_validation = validation_ds.map(tokenize, batched=True)
    return tokenized_train, tokenized_validation


def tokenize(batch):
    tokenized = tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=512
    )
    tokenized["labels"] = batch["is_fake"]
    return tokenized


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train(train_phase: TrainPhase):
    train_ds, validation_ds = get_tokenized_dataset(train_phase)
    output_dir, logging_dir = Utils.get_model_train_output_path(train_phase)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=TRAIN_EPOCHS,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=42,
        logging_strategy="epoch",
        report_to=["tensorboard"],
        logging_dir=logging_dir,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    args = Utils.parse_train_phase_args()
    train_phase = args.phase
    load_tokenizer_and_model(train_phase)
    train(train_phase)
