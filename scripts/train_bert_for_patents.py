import argparse
import numpy as np

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, default="data/processed/train.csv")
    parser.add_argument("--valid_file", type=str, default="data/processed/valid.csv")

    # Real BERT for Patents model.
    parser.add_argument("--model_name", type=str, default="anferico/bert-for-patents")

    parser.add_argument("--output_dir", type=str, default="outputs/bert_for_patents_baseline")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    return parser.parse_args()


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = softmax(logits)
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)

    # Positive class: label = 1 = novelty-destroying prior art.
    destroying_precision, destroying_recall, destroying_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    # Negative class: label = 0 = not novelty-destroying prior art.
    background_precision, background_recall, background_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        pos_label=0,
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "destroying_precision": destroying_precision,
        "destroying_recall": destroying_recall,
        "destroying_f1": destroying_f1,
        "background_precision": background_precision,
        "background_recall": background_recall,
        "background_f1": background_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "auc": auc,
    }


def main():
    args = parse_args()

    print(f"Loading dataset:")
    print(f"  train: {args.train_file}")
    print(f"  valid: {args.valid_file}")

    dataset = load_dataset(
        "csv",
        data_files={
            "train": args.train_file,
            "validation": args.valid_file,
        },
    )

    print(f"Loading tokenizer from: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    def tokenize_examples(examples):
        return tokenizer(
            examples["patent_application"],
            examples["prior_art"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_examples,
        batched=True,
        remove_columns=["patent_application", "prior_art"],
    )

    tokenized_dataset = tokenized_dataset.rename_column("novelty", "labels")

    print(f"Loading model from: {args.model_name}")

    config = BertConfig.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="destroying_f1",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("Final validation evaluation:")
    print(trainer.evaluate())

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved model and tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()