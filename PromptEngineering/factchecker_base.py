#!/usr/bin/env python3
"""
factchecker_base.py

This module provides the base API and shared functionality for table fact checking.
It includes utility functions, common classes, metrics plotting, parallel processing,
and error analysis functions. All approaches (prompt engineering, RAG, code generation)
should import and use these functions.
"""

import os
import json
import logging
import re
import gc
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    auc,
)

# Import your LLM interface.
from langchain_ollama import OllamaLLM
import torch

################################################################################
#                             LOGGING CONFIGURATION
################################################################################

log_filename = f"logs/logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode="w")
    ],
)

################################################################################
#                          UTILITY FUNCTIONS
################################################################################

def merge_training_data(file1: str, file2: str, output_file: str) -> None:
    """
    Merge two JSON files containing training data and save the combined result.
    """
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            results1 = json.load(f1)
            results2 = json.load(f2)
        results2.update(results1)
        with open(output_file, 'w') as f_out:
            json.dump(results2, f_out, indent=2)
        logging.info(f"Successfully merged {file1} and {file2} into {output_file}.")
    except Exception as e:
        logging.error(f"Error merging files {file1} and {file2}: {e}")

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file from disk.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logging.info(f"Loaded data from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"Failed to load JSON file {file_path}: {e}")
        return {}

################################################################################
#                        TABLE FORMAT CONVERTERS
################################################################################

def format_table_to_markdown(table: pd.DataFrame) -> str:
    """Convert a pandas DataFrame to a Markdown-formatted string."""
    return table.to_markdown(index=False)

def naturalize_table(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to a naturalized text format.
    """
    rows = []
    for row_idx, row in df.iterrows():
        row_description = [f"Row {row_idx + 1} is:"]
        for col_name, cell_value in row.items():
            row_description.append(f"{col_name} is {cell_value}")
        rows.append(" ".join(row_description) + ".")
    return " ".join(rows)

################################################################################
#                         RESPONSE EXTRACTION
################################################################################

def extract_json_from_response(raw_response: str) -> dict:
    """
    Try to extract JSON from a model response by first searching for a JSON code fence,
    and then by attempting to parse the entire response.
    """
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.warning(f"[WARN] JSON decode error (inside code fence): {e}")
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        logging.warning("[WARN] Could not parse raw_response as JSON either.")
        return {}

################################################################################
#                         BASE FACT CHECKER CLASS
################################################################################

class BaseFactChecker:
    """
    Base class for table fact checking.
    
    Provides common functionality:
      - Loading and formatting tables
      - Invoking the LLM
      - Parsing the LLM output
      - Processing a single claim
      
    Subclasses must implement the generate_prompt() method.
    """
    def __init__(
        self,
        all_csv_folder: str,
        learning_type: str = "zero_shot",
        format_type: str = "naturalized",
        model: Optional[OllamaLLM] = None
    ):
        self.all_csv_folder = all_csv_folder
        self.learning_type = learning_type
        self.format_type = format_type
        self.model = model

    def load_table(self, table_id: str) -> Optional[pd.DataFrame]:
        """
        Load a table from a CSV file given its ID.
        """
        table_path = os.path.join(self.all_csv_folder, table_id)
        try:
            table = pd.read_csv(table_path, delimiter="#")
            logging.debug(f"Loaded table {table_id} from {table_path}.")
            return table
        except Exception as e:
            logging.error(f"Failed to load table {table_id} from {table_path}: {e}")
            return None

    def format_table(self, table: pd.DataFrame) -> str:
        """
        Format the table using the specified format (markdown or naturalized).
        """
        if self.format_type == "markdown":
            return format_table_to_markdown(table)
        else:
            return naturalize_table(table)

    def invoke_llm(self, prompt: str) -> str:
        """
        Invoke the LLM with the given prompt.
        """
        if not self.model:
            raise ValueError("LLM model not initialized.")
        try:
            return self.model.invoke(prompt).strip()
        except Exception as e:
            logging.error(f"Error invoking LLM: {e}")
            return ""

    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract the JSON output.
        """
        try:
            parsed_dict = extract_json_from_response(raw_response)
            answer = parsed_dict.get("answer", "FALSE").upper()
            relevant_cells = parsed_dict.get("relevant_cells", [])
        except Exception as e:
            logging.warning(f"Error parsing response: {e}. Raw response: {raw_response}")
            answer = "FALSE"
            relevant_cells = []
        return {"answer": answer, "relevant_cells": relevant_cells}

    def process_claim(self, table_id: str, claim: str, true_label: int) -> Dict[str, Any]:
        """
        Process a single claim: generate prompt, invoke LLM, parse response,
        and return a result dictionary.
        """
        prompt = self.generate_prompt(table_id, claim)
        if prompt is None:
            logging.warning(f"Prompt generation failed for table {table_id}, claim: {claim}")
            return {
                "table_id": table_id,
                "claim": claim,
                "predicted_response": None,
                "resp": "PROMPT_ERROR",
                "true_response": true_label,
                "relevant_cells": []
            }
        raw_response = self.invoke_llm(prompt)
        parsed = self.parse_response(raw_response)
        predicted_label = 1 if parsed.get("answer") == "TRUE" else 0
        return {
            "table_id": table_id,
            "claim": claim,
            "predicted_response": predicted_label,
            "resp": raw_response,
            "true_response": true_label,
            "relevant_cells": parsed.get("relevant_cells")
        }

    def generate_prompt(self, table_id: str, claim: str) -> Optional[str]:
        """
        Abstract method for generating a prompt.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_prompt()")

################################################################################
#                      TESTING & METRICS FUNCTIONS
################################################################################

def test_model_on_claims(
    fact_checker: BaseFactChecker,
    full_cleaned_data: Dict[str, Any],
    test_all: bool = False,
    N: int = 10,
    checkpoint_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Test the fact-checking model on the provided dataset.
    """
    results = []
    completed_keys = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        logging.info(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, "r") as f:
            results = json.load(f)
        for row in results:
            completed_keys.add((row["table_id"], row["claim"]))
    keys = list(full_cleaned_data.keys())
    limit = len(keys) if test_all else min(N, len(keys))
    logging.info(f"Testing {limit} tables out of {len(keys)}.")
    for i in tqdm(range(limit), desc="Processing tables"):
        table_id = keys[i]
        claims = full_cleaned_data[table_id][0]
        labels = full_cleaned_data[table_id][1]
        for idx, claim in enumerate(claims):
            if (table_id, claim) in completed_keys:
                continue
            result = fact_checker.process_claim(table_id, claim, labels[idx])
            results.append(result)
            completed_keys.add((table_id, claim))
            if checkpoint_file:
                with open(checkpoint_file, "w") as f:
                    json.dump(results, f, indent=2)
    return results

def process_claim_worker(
    fact_checker_class,
    model_name: str,
    table_id: str,
    claim: str,
    true_label: int,
    all_csv_folder: str,
    learning_type: str,
    format_type: str,
    approach: str,
) -> Dict[str, Any]:
    """
    Worker function for parallel processing.
    Initializes its own LLM and fact checker instance.
    """
    try:
        if "deepseek" in model_name:
            model = OllamaLLM(model=model_name, params={"n_ctx": 4096, "n_batch": 256})
        else:
            model = OllamaLLM(model=model_name)
        fact_checker = fact_checker_class(all_csv_folder, learning_type, format_type, model)
        return fact_checker.process_claim(table_id, claim, true_label)
    except Exception as e:
        logging.error(f"Error processing claim for table {table_id}, claim: {claim}. Error: {e}")
        return {
            "table_id": table_id,
            "claim": claim,
            "predicted_response": None,
            "resp": "PROCESSING_ERROR",
            "true_response": true_label,
            "relevant_cells": []
        }

def test_model_on_claims_parallel(
    fact_checker_class,  # The class (subclass of BaseFactChecker)
    model_name: str,
    full_cleaned_data: Dict[str, Any],
    all_csv_folder: str,
    learning_type: str,
    format_type: str,
    approach: str,
    test_all: bool = False,
    N: int = 10,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Test the fact-checking model in parallel across multiple processes.
    """
    results = []
    keys = list(full_cleaned_data.keys())
    limit = len(keys) if test_all else min(N, len(keys))
    logging.info(f"Testing {limit} tables out of {len(keys)} in parallel.")
    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(limit):
            table_id = keys[i]
            claims = full_cleaned_data[table_id][0]
            labels = full_cleaned_data[table_id][1]
            for idx, claim in enumerate(claims):
                tasks.append(executor.submit(
                    process_claim_worker,
                    fact_checker_class,
                    model_name,
                    table_id,
                    claim,
                    labels[idx],
                    all_csv_folder,
                    learning_type,
                    format_type,
                    approach
                ))
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing claims"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Error in parallel processing: {e}")
    return results

################################################################################
#                           METRICS PLOTTING
################################################################################

def plot_confusion_matrix_plot(
    y_true: List[int],
    y_pred: List[int],
    classes: List[int],
    title: str,
    save_path: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.debug(f"Saved confusion matrix plot to {save_path}.")

def plot_roc_curve_plot(
    y_true: List[int],
    y_pred: List[int],
    title: str,
    save_path: str,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.debug(f"Saved ROC curve plot to {save_path}.")

def calculate_and_plot_metrics(
    results: List[Dict[str, Any]],
    save_dir: str = "results_plots",
    save_stats_file: str = "summary_stats.json",
    learning_type: str = "",
    dataset_type: str = "",
    model_name: str = "",
    format_type: str = "naturalized",
) -> None:
    y_true = [r["true_response"] for r in results if r["predicted_response"] is not None]
    y_pred = [r["predicted_response"] for r in results if r["predicted_response"] is not None]
    classes = [0, 1]
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Saving metrics to directory: {save_dir}")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0, 0] if cm.shape[0] > 0 else 0
        fp = cm[0, 1] if cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        tp = cm[1, 1] if cm.shape[1] > 1 and cm.shape[0] > 1 else 0

    cm_title = f"Confusion Matrix - {learning_type.capitalize()} - {dataset_type.capitalize()}"
    cm_save_path = os.path.join(save_dir, f"confusion_matrix_{learning_type}_{dataset_type}.png")
    plot_confusion_matrix_plot(y_true, y_pred, classes, cm_title, cm_save_path)

    roc_title = f"ROC Curve - {learning_type.capitalize()} - {dataset_type.capitalize()}"
    roc_save_path = os.path.join(save_dir, f"roc_curve_{learning_type}_{dataset_type}.png")
    plot_roc_curve_plot(y_true, y_pred, roc_title, roc_save_path)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    stats = {
        "model_name": model_name,
        "learning_type": learning_type,
        "dataset_type": dataset_type,
        "format_type": format_type,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        else:
            return obj
    stats_save_path = os.path.join(save_dir, save_stats_file)
    try:
        serializable_stats = convert_to_serializable(stats)
        with open(stats_save_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)
        logging.info(f"Saved summary statistics to {stats_save_path}.")
    except Exception as e:
        logging.error(f"Failed to save summary statistics: {e}")
    logging.info(f"Metrics:\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}\nAccuracy: {accuracy:.2f}")

################################################################################
#                             ERROR ANALYSIS
################################################################################

def analyze_errors(result_file: str) -> None:
    """
    Analyze errors from a result JSON file and print a summary.
    """
    if not os.path.exists(result_file):
        logging.error(f"Result file {result_file} does not exist.")
        return
    try:
        with open(result_file, "r") as f:
            results = json.load(f)
    except Exception as e:
        logging.error(f"Error loading result file {result_file}: {e}")
        return
    error_counts = {}
    for r in results:
        if r["predicted_response"] is None or r["resp"] in ["PROMPT_ERROR", "PROCESSING_ERROR"]:
            error_counts.setdefault(r["resp"], 0)
            error_counts[r["resp"]] += 1
    logging.info("Error Analysis Summary:")
    for err, count in error_counts.items():
        logging.info(f"{err}: {count}")

def load_and_print_stats(json_file_path: str) -> None:
    """
    Load and print summary statistics from a JSON file.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} does not exist.")
        return
    try:
        with open(json_file_path, "r") as f:
            stats = json.load(f)
        print("Summary Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error reading stats file: {e}")
