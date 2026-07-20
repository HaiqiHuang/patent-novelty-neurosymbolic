import argparse
import os

import numpy as np
import torch

from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed_v3_all_parent/train.csv",
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        default="data/processed_v3_all_parent/valid.csv",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen3_0.6b_all_parents_2048_lora",
    )

    parser.add_argument(
        "--claim_column",
        type=str,
        default="patent_application",
    )
    parser.add_argument(
        "--prior_art_column",
        type=str,
        default="prior_art",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="novelty",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max_prior_art_length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--use_bf16",
        action="store_true",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
    )

    return parser.parse_args()


def softmax(logits):
    exp_logits = np.exp(
        logits - np.max(logits, axis=1, keepdims=True)
    )
    return exp_logits / exp_logits.sum(
        axis=1,
        keepdims=True,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Some model outputs may arrive wrapped in a tuple.
    if isinstance(logits, tuple):
        logits = logits[0]

    probs = softmax(logits)
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)

    (
        destroying_precision,
        destroying_recall,
        destroying_f1,
        _,
    ) = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    (
        background_precision,
        background_recall,
        background_f1,
        _,
    ) = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        pos_label=0,
        zero_division=0,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(
            labels,
            probs[:, 1],
        )
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


def get_torch_dtype(args):
    if args.use_bf16:
        return torch.bfloat16

    if args.use_fp16:
        return torch.float16

    return torch.float32


def main():
    args = parse_args()

    if args.use_bf16 and args.use_fp16:
        raise ValueError(
            "Choose only one of --use_bf16 or --use_fp16."
        )

    if args.max_prior_art_length >= args.max_length:
        raise ValueError(
            "--max_prior_art_length must be smaller "
            "than --max_length."
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    set_seed(args.seed)

    print("=" * 80)
    print("Loading dataset")
    print("=" * 80)
    print("Train file:", args.train_file)
    print("Validation file:", args.valid_file)
    print("Model:", args.model_name)
    print("Max length:", args.max_length)

    dataset = load_dataset(
        "csv",
        data_files={
            "train": args.train_file,
            "validation": args.valid_file,
        },
    )

    required_columns = {
        args.claim_column,
        args.prior_art_column,
        args.label_column,
    }

    available_columns = set(
        dataset["train"].column_names
    )

    missing_columns = (
        required_columns - available_columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. "
            f"Available columns: {available_columns}"
        )

    print("Train rows:", len(dataset["train"]))
    print(
        "Validation rows:",
        len(dataset["validation"]),
    )
    print(
        "CSV columns:",
        dataset["train"].column_names,
    )

    print("=" * 80)
    print("Loading tokenizer")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )

    # Decoder-only models may not define an independent pad token.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    claim_prefix_ids = tokenizer(
        "[ENRICHED CLAIM]\n",
        add_special_tokens=False,
    )["input_ids"]

    separator_ids = tokenizer(
        "\n\n[PRIOR ART]\n",
        add_special_tokens=False,
    )["input_ids"]

    eos_ids = []

    if tokenizer.eos_token_id is not None:
        eos_ids = [tokenizer.eos_token_id]

    fixed_token_count = (
        len(claim_prefix_ids)
        + len(separator_ids)
        + len(eos_ids)
    )

    def tokenize_examples(examples):
        all_input_ids = []
        all_attention_masks = []
        all_was_truncated = []
        all_original_lengths = []

        claims = examples[args.claim_column]
        prior_arts = examples[args.prior_art_column]

        for claim, prior_art in zip(
            claims,
            prior_arts,
        ):
            claim = "" if claim is None else str(claim)
            prior_art = (
                ""
                if prior_art is None
                else str(prior_art)
            )

            # Tokenize prior art separately so that long enriched claims
            # cannot remove the prior-art evidence.
            full_prior_art_ids = tokenizer(
                prior_art,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            prior_art_ids = full_prior_art_ids[
                : args.max_prior_art_length
            ]

            claim_budget = (
                args.max_length
                - fixed_token_count
                - len(prior_art_ids)
            )

            if claim_budget <= 0:
                raise ValueError(
                    "No token budget remains for the enriched claim. "
                    "Reduce --max_prior_art_length or increase "
                    "--max_length."
                )

            full_claim_ids = tokenizer(
                claim,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            claim_ids = full_claim_ids[:claim_budget]

            input_ids = (
                claim_prefix_ids
                + claim_ids
                + separator_ids
                + prior_art_ids
                + eos_ids
            )

            # Safety guard.
            input_ids = input_ids[: args.max_length]

            original_length = (
                fixed_token_count
                + len(full_claim_ids)
                + len(full_prior_art_ids)
            )

            was_truncated = (
                len(full_claim_ids) > len(claim_ids)
                or len(full_prior_art_ids)
                > len(prior_art_ids)
            )

            all_input_ids.append(input_ids)
            all_attention_masks.append(
                [1] * len(input_ids)
            )
            all_original_lengths.append(
                original_length
            )
            all_was_truncated.append(
                int(was_truncated)
            )

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "original_token_length": all_original_lengths,
            "was_truncated": all_was_truncated,
        }

    tokenized_dataset = dataset.map(
        tokenize_examples,
        batched=True,
        desc="Tokenizing Qwen inputs",
    )

    tokenized_dataset = (
        tokenized_dataset.rename_column(
            args.label_column,
            "labels",
        )
    )

    # Trainer only needs model inputs and labels.
    keep_columns = {
        "input_ids",
        "attention_mask",
        "labels",
    }

    remove_columns = [
        column
        for column in tokenized_dataset[
            "train"
        ].column_names
        if column not in keep_columns
    ]

    tokenized_for_training = (
        tokenized_dataset.remove_columns(
            remove_columns
        )
    )

    first_debug_example = (
        tokenized_dataset["train"][0]
    )

    print("=" * 80)
    print("Tokenization sanity check")
    print("=" * 80)
    print(
        "Training columns:",
        tokenized_for_training[
            "train"
        ].column_names,
    )
    print(
        "First input length:",
        len(first_debug_example["input_ids"]),
    )
    print(
        "First original length:",
        first_debug_example[
            "original_token_length"
        ],
    )
    print(
        "First was truncated:",
        bool(
            first_debug_example[
                "was_truncated"
            ]
        ),
    )
    print(
        "First label:",
        first_debug_example["labels"],
    )
    print("First decoded input:")
    print(
        tokenizer.decode(
            first_debug_example["input_ids"],
            skip_special_tokens=False,
        )[:5000]
    )

    train_truncation_rate = np.mean(
        tokenized_dataset["train"][
            "was_truncated"
        ]
    )
    valid_truncation_rate = np.mean(
        tokenized_dataset["validation"][
            "was_truncated"
        ]
    )

    print(
        f"Train truncation rate: "
        f"{train_truncation_rate:.2%}"
    )
    print(
        f"Validation truncation rate: "
        f"{valid_truncation_rate:.2%}"
    )

    print("=" * 80)
    print("Loading Qwen sequence classifier")
    print("=" * 80)

    dtype = get_torch_dtype(args)

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.model_name,
            num_labels=2,
            torch_dtype=dtype,
        )
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.config.use_cache = False

    model.config.id2label = {
        0: "background",
        1: "destroying",
    }
    model.config.label2id = {
        "background": 0,
        "destroying": 1,
    }

    print("Loaded model class:", type(model))
    print("Classification-related modules:")

    for name, module in model.named_modules():
        if (
            name == "score"
            or name.endswith(".score")
            or "classifier" in name
        ):
            print(name, module)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        # Qwen3ForSequenceClassification uses "score".
        modules_to_save=["score"],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        learning_rate=args.learning_rate,
        per_device_train_batch_size=(
            args.train_batch_size
        ),
        per_device_eval_batch_size=(
            args.eval_batch_size
        ),
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_strategy="steps",
        logging_steps=args.logging_steps,

        load_best_model_at_end=True,

        # You previously used destroying_f1.
        # For balanced binary data, macro_f1 is usually
        # a safer overall checkpoint criterion.
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        bf16=args.use_bf16,
        fp16=args.use_fp16,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        seed=args.seed,
        data_seed=args.seed,

        dataloader_num_workers=(
            args.num_workers
        ),

        save_total_limit=2,
        report_to="none",

        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_for_training[
            "train"
        ],
        eval_dataset=tokenized_for_training[
            "validation"
        ],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("=" * 80)
    print("Starting training")
    print("=" * 80)

    trainer.train()

    print("=" * 80)
    print("Final validation evaluation")
    print("=" * 80)

    metrics = trainer.evaluate()

    for key, value in metrics.items():
        print(f"{key}: {value}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(
        args.output_dir
    )

    print(
        f"Saved model and tokenizer to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()