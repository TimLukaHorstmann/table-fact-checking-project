# run_tabfact_with_tapex.py (function-based)
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    BartForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TapexTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from torch.nn.functional import softmax

# Will error if the minimal version of Transformers is not installed.
check_min_version("4.17.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="tab_fact", 
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default="tab_fact",
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization."
                "Sequences longer than this will be truncated."
            )
        },
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."})
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad dynamically to the max length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples if set."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of evaluation examples if set."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of prediction examples if set."},
    )
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the test data."}
    )

    def __post_init__(self):
        # Original checks for dataset or local file usage
        if self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a dataset or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be csv or json."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension as `train_file`."


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use a fast tokenizer backed by the tokenizers library or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (branch name, tag name, or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token from `huggingface-cli login` if needed for private models."},
    )


def run_tabfact_experiment(args_dict: Dict[str, Any]):
    """
    Function-based approach to run TabFact with TAPEX, allowing calls from a notebook.
    Returns a dictionary with metrics and predictions so you can plot confusion matrices, etc.

    Example usage:

        from run_tabfact_adjusted import run_tabfact_experiment

        args = {
            "do_eval": True,
            "model_name_or_path": "microsoft/tapex-base-finetuned-tabfact",
            "output_dir": "tabfact_tapex_base_eval",
            "per_device_eval_batch_size": 12,
            "eval_accumulation_steps": 6,
            "cache_dir": "/Data/tlh45",
            # etc...
        }

        results = run_tabfact_experiment(args)
        if results is not None:
            print("Evaluation results:", results["eval_results"])
            labels = results["labels"]
            probs = results["probs"]
            preds = results["preds"]
            # create plots, etc.
    """

    # -------------------
    # 1. Convert the dict into HF's argument classes or direct usage
    # -------------------
    from transformers import TrainingArguments

    # Provide defaults for missing arguments in the dictionary
    args_dict.setdefault("do_train", False)
    args_dict.setdefault("do_eval", False)
    args_dict.setdefault("do_predict", False)
    args_dict.setdefault("dataset_name", "tab_fact")
    args_dict.setdefault("dataset_config_name", "tab_fact")
    args_dict.setdefault("max_seq_length", 1024)
    args_dict.setdefault("overwrite_cache", False)
    args_dict.setdefault("pad_to_max_length", False)
    args_dict.setdefault("train_file", None)
    args_dict.setdefault("validation_file", None)
    args_dict.setdefault("test_file", None)
    args_dict.setdefault("model_name_or_path", None)
    args_dict.setdefault("cache_dir", None)
    args_dict.setdefault("use_fast_tokenizer", True)
    args_dict.setdefault("model_revision", "main")
    args_dict.setdefault("use_auth_token", False)
    args_dict.setdefault("output_dir", "outputs")
    args_dict.setdefault("per_device_eval_batch_size", 16)
    args_dict.setdefault("eval_accumulation_steps", 1)
    args_dict.setdefault("per_device_train_batch_size", 8)
    args_dict.setdefault("seed", 42)
    args_dict.setdefault("fp16", False)
    args_dict.setdefault("local_rank", -1)
    args_dict.setdefault("overwrite_output_dir", False)

    # We'll create model_args, data_args, and training_args from these
    model_args = ModelArguments(
        model_name_or_path=args_dict["model_name_or_path"],
        config_name=args_dict.get("config_name", None),
        tokenizer_name=args_dict.get("tokenizer_name", None),
        cache_dir=args_dict["cache_dir"],
        use_fast_tokenizer=args_dict["use_fast_tokenizer"],
        model_revision=args_dict["model_revision"],
        use_auth_token=args_dict["use_auth_token"],
    )
    data_args = DataTrainingArguments(
        dataset_name=args_dict["dataset_name"],
        dataset_config_name=args_dict["dataset_config_name"],
        max_seq_length=args_dict["max_seq_length"],
        overwrite_cache=args_dict["overwrite_cache"],
        pad_to_max_length=args_dict["pad_to_max_length"],
        train_file=args_dict["train_file"],
        validation_file=args_dict["validation_file"],
        test_file=args_dict["test_file"],
        # handle optional max_{train,eval,predict}_samples if needed
        max_train_samples=args_dict.get("max_train_samples", None),
        max_eval_samples=args_dict.get("max_eval_samples", None),
        max_predict_samples=args_dict.get("max_predict_samples", None),
    )
    training_args = TrainingArguments(
        output_dir=args_dict["output_dir"],
        do_train=args_dict["do_train"],
        do_eval=args_dict["do_eval"],
        do_predict=args_dict.get("do_predict", False),
        per_device_train_batch_size=args_dict.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=args_dict["per_device_eval_batch_size"],
        eval_accumulation_steps=args_dict["eval_accumulation_steps"],
        overwrite_output_dir=args_dict["overwrite_output_dir"],
        seed=args_dict["seed"],
        fp16=args_dict["fp16"],
        local_rank=args_dict["local_rank"],
    )

    # Setup logging
    log_level = training_args.get_process_log_level()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detect last checkpoint (if resuming training)
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "Change `--output_dir` or add `--overwrite_output_dir` to start from scratch."
            )

    # Set seed
    set_seed(training_args.seed)

    # -------------------
    # 2. Load the dataset
    # -------------------
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True,  # required for Tapex
        )
    else:
        data_files = {}
        if args_dict["do_train"]:
            data_files["train"] = data_args.train_file
        if args_dict["do_eval"]:
            data_files["validation"] = data_args.validation_file
        if args_dict.get("do_predict", False):
            data_files["test"] = data_args.test_file

        raw_datasets = load_dataset(
            "csv" if data_args.train_file.endswith(".csv") else "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir
        )

    # We assume we have "label" with names, typical tabfact data
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    # -------------------
    # 3. Load model & tokenizer
    # -------------------
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
    )
    tokenizer = TapexTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
    )
    model = BartForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
    )

    # Overwrite label mappings for TabFact
    model.config.label2id = {"Refused": 0, "Entailed": 1}
    model.config.id2label = {0: "Refused", 1: "Entailed"}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"max_seq_length ({data_args.max_seq_length}) is larger than model_max_length ({tokenizer.model_max_length}). "
            f"Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Decide on padding approach
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    # -------------------
    # 4. Preprocessing
    # -------------------
    def _convert_table_text_to_pandas(table_text: str):
        """Converts string-based table_text to a pandas DataFrame."""
        # Example table_text might be: "round#clubs remaining\nfirst round#156\n"
        lines = table_text.strip("\n").split("\n")
        rows = [row.split("#") for row in lines]
        df = pd.DataFrame.from_records(rows[1:], columns=rows[0])
        return df

    def preprocess_tabfact_function(examples):
        questions = examples["statement"]
        tables = [ _convert_table_text_to_pandas(t) for t in examples["table_text"] ]
        result = tokenizer(
            tables,
            questions,
            padding=padding,
            max_length=max_seq_length,
            truncation=True
        )
        result["label"] = examples["label"]
        return result

    # Map the dataset
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_tabfact_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # Prepare train/eval/predict splits
    train_dataset, eval_dataset, predict_dataset = None, None, None

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict and "test" in raw_datasets:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log random samples
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Compute metrics
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # -------------------
    # 5. Training
    # -------------------
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # -------------------
    # 6. Evaluation
    # -------------------
    result_dict = None
    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # Additionally, we do `trainer.predict` on eval_dataset to get raw predictions
        eval_preds = trainer.predict(eval_dataset)
        raw_logits = eval_preds.predictions
        label_ids = eval_preds.label_ids

        # Convert raw logits to predicted probabilities for the 'Entailed' label=1
        # If you prefer the entire probability vector, store that too.
        # For 2-class classification, shape = (num_examples, 2)
        # We'll store probability for label=1
        probs = softmax(raw_logits, axis=1)[:, 1]
        preds = np.argmax(raw_logits, axis=1)

        # Save to dictionary so you can plot confusion matrices, etc.
        # Ground truth (label_ids) is numeric, 0 or 1. 
        # Return them as-is, or convert them to 0/1 for the user.
        result_dict = {
            "eval_results": metrics,        # e.g. {"accuracy": 0.85, ...}
            "labels": label_ids.tolist(),   # ground-truth labels
            "probs": probs.tolist(),        # predicted probability of label=1
            "preds": preds.tolist()         # predicted label, e.g. 0 or 1
        }

    # -------------------
    # 7. Prediction (test)
    # -------------------
    if training_args.do_predict and predict_dataset is not None:
        logger.info("*** Predict ***")
        # Remove the `label` column if it exists to avoid issues
        if "label" in predict_dataset.column_names:
            predict_dataset = predict_dataset.remove_columns("label")

        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, "predict_results_tabfact.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Predict Results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")

    # Return the dictionary with eval results (labels, probs, preds) if do_eval was done
    return result_dict


def _mp_fn(index):
    # For xla_spawn (TPUs)
    pass


if __name__ == "__main__":
    # The main() below is no longer used in the same way,
    # because we want function-based usage from a notebook.
    # But we keep an entry point if you DO want to run it as a script.
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Convert to dictionary
    args_dict = {}
    # Merge model_args
    args_dict["model_name_or_path"] = model_args.model_name_or_path
    args_dict["config_name"] = model_args.config_name
    args_dict["tokenizer_name"] = model_args.tokenizer_name
    args_dict["cache_dir"] = model_args.cache_dir
    args_dict["use_fast_tokenizer"] = model_args.use_fast_tokenizer
    args_dict["model_revision"] = model_args.model_revision
    args_dict["use_auth_token"] = model_args.use_auth_token
    # Merge data_args
    args_dict["dataset_name"] = data_args.dataset_name
    args_dict["dataset_config_name"] = data_args.dataset_config_name
    args_dict["max_seq_length"] = data_args.max_seq_length
    args_dict["overwrite_cache"] = data_args.overwrite_cache
    args_dict["pad_to_max_length"] = data_args.pad_to_max_length
    args_dict["train_file"] = data_args.train_file
    args_dict["validation_file"] = data_args.validation_file
    args_dict["test_file"] = data_args.test_file
    args_dict["max_train_samples"] = data_args.max_train_samples
    args_dict["max_eval_samples"] = data_args.max_eval_samples
    args_dict["max_predict_samples"] = data_args.max_predict_samples
    # Merge training_args
    args_dict["output_dir"] = training_args.output_dir
    args_dict["do_train"] = training_args.do_train
    args_dict["do_eval"] = training_args.do_eval
    args_dict["do_predict"] = training_args.do_predict
    args_dict["per_device_train_batch_size"] = training_args.per_device_train_batch_size
    args_dict["per_device_eval_batch_size"] = training_args.per_device_eval_batch_size
    args_dict["eval_accumulation_steps"] = training_args.eval_accumulation_steps
    args_dict["overwrite_output_dir"] = training_args.overwrite_output_dir
    args_dict["seed"] = training_args.seed
    args_dict["fp16"] = training_args.fp16
    args_dict["local_rank"] = training_args.local_rank

    run_tabfact_experiment(args_dict)