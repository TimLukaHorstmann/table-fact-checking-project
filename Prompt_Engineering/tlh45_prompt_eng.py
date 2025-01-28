from datetime import datetime
import os
import sys
import json
import pickle
import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache
import argparse
import numpy as np
import re

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    auc,
)
from langchain_ollama import OllamaLLM
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import torch

# Configure logging
log_filename = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler(log_filename, mode="w")  # Creates a new log file
    ],
)

################################################################################
#                           PROMPT TEMPLATES
################################################################################

# ------------------------------------------------------------------------------
# UPDATED END_OF_PROMPT_INSTRUCTIONS: 
# We instruct the model to produce a single JSON with two fields:
#   "answer" : "TRUE" or "FALSE"
#   "highlighted_cells": a list of row/col references
# ------------------------------------------------------------------------------
END_OF_PROMPT_INSTRUCTIONS = """
Return a valid JSON object with two keys:
"answer": must be "TRUE" or "FALSE" (all caps)
"highlighted_cells": a list of objects, each with "row_index" (int) and "column_name" (string)

For example:

{{
  "answer": "TRUE",
  "highlighted_cells": [
    {{"row_index": 0, "column_name": "Revenue"}},
    {{"row_index": 1, "column_name": "Employees"}}
  ]
}}

No extra keys, no extra text. Just that JSON.
"""

# -------------------------- ZERO-SHOT PROMPTS --------------------------
ZERO_SHOT_PROMPT_MARKDOWN = """
You are tasked with determining whether a claim about the following table (in Markdown) is TRUE or FALSE.

Table (Markdown):
{table_formatted}

Claim: "{claim}"

Instructions:
- Carefully check each condition in the claim against the table.
- If fully supported, the 'answer' should be "TRUE". Otherwise "FALSE".
""" + END_OF_PROMPT_INSTRUCTIONS

ZERO_SHOT_PROMPT_NATURALIZED = """
You are tasked with determining whether a claim about the following table (in natural text) is TRUE or FALSE.

Table (Naturalized):
{table_formatted}

Claim: "{claim}"

Instructions:
- If the claim is supported by the data, 'answer' should be "TRUE".
- Otherwise, 'answer' is "FALSE".
""" + END_OF_PROMPT_INSTRUCTIONS


# -------------------------- ONE-SHOT PROMPTS --------------------------
ONE_SHOT_EXAMPLE_MARKDOWN = """
Example of evaluating a claim with a Markdown table:

| Product | Price | Stock |
|---------|-------|-------|
| A       | 10    | 100   |
| B       | 15    | 50    |

Claim: "Product B is cheaper than Product A."

Analysis and Answer:
- Product A's price is 10, Product B's price is 15.
- The claim says B is cheaper, but that's incorrect, B is more expensive.
- So the correct answer is FALSE.
"""

ONE_SHOT_PROMPT_MARKDOWN = """
Below is an example of how to evaluate a claim using a table in Markdown:

{one_shot_example}

Now, evaluate the following table and claim in the same manner.

Table (Markdown):
{table_formatted}

Claim: "{claim}"

Instructions:
- Compare the claim to the table data. If fully supported, 'answer' is "TRUE", else "FALSE".

""" + END_OF_PROMPT_INSTRUCTIONS

ONE_SHOT_EXAMPLE_NATURALIZED = """
Example of evaluating a claim with a table in naturalized text:

Row 1: Product is A, Price is 10, Stock is 100.
Row 2: Product is B, Price is 15, Stock is 50.

Claim: "Product A is the most expensive item."

Analysis and Answer:
- A costs 10, B costs 15.
- B is more expensive, so the claim is FALSE.
"""

ONE_SHOT_PROMPT_NATURALIZED = """
Below is an example of how to evaluate a claim using a table in naturalized text:

{one_shot_example}

Now, evaluate the following table and claim in the same manner.

Table (Naturalized):
{table_formatted}

Claim: "{claim}"

Instructions:
- If the claim is fully supported by the data, answer "TRUE"; otherwise "FALSE".

""" + END_OF_PROMPT_INSTRUCTIONS


# -------------------------- FEW-SHOT PROMPTS --------------------------
FEW_SHOT_EXAMPLE_MARKDOWN = """
Example 1:
Table (Markdown):
| Country | Population |
|---------|-----------|
| X       | 10        |
| Y       | 20        |

Claim: "Country X has a larger population than Country Y."
Answer: FALSE (Because 10 < 20).

Example 2:
Table (Markdown):
| Company | Revenue | Employees |
|---------|---------|-----------|
| A       | 100     | 500       |
| B       | 200     | 1000      |

Claim: "Company A has fewer employees than Company B."
Answer: TRUE (Because 500 < 1000).
"""

FEW_SHOT_PROMPT_MARKDOWN = """
You will see multiple examples of how to evaluate a claim using Markdown tables:

{few_shot_example}

Now, evaluate a new table and claim.

Table (Markdown):
{table_formatted}

Claim: "{claim}"

Instructions:
- Determine if the claim is supported (TRUE) or not (FALSE).

""" + END_OF_PROMPT_INSTRUCTIONS

FEW_SHOT_EXAMPLE_NATURALIZED = """
Example 1 (Naturalized):
Row 1: Country is X, Population is 10
Row 2: Country is Y, Population is 20

Claim: "Country X has a larger population than Country Y."
Answer: FALSE (10 < 20).

Example 2 (Naturalized):
Row 1: Company is A, Revenue is 100, Employees is 500
Row 2: Company is B, Revenue is 200, Employees is 1000

Claim: "Company B has more revenue than Company A."
Answer: TRUE (200 > 100).
"""

FEW_SHOT_PROMPT_NATURALIZED = """
You will see multiple examples of how to evaluate a claim using naturalized text:

{few_shot_example}

Now, evaluate a new table and claim.

Table (Naturalized):
{table_formatted}

Claim: "{claim}"

Instructions:
- Check if the data supports the claim. If yes, 'answer' is "TRUE"; otherwise "FALSE".

""" + END_OF_PROMPT_INSTRUCTIONS


################################################################################
#                           MERGE FUNCTION
################################################################################

def merge_training_data(file1: str, file2: str, output_file: str) -> None:
    """
    Merge two JSON files containing training data and save the combined result.

    Args:
        file1 (str): Path to the first JSON file.
        file2 (str): Path to the second JSON file.
        output_file (str): Path to the output JSON file where merged data will be saved.
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


################################################################################
#                           LOAD TABLE
################################################################################

@lru_cache(maxsize=None)
def load_table(all_csv_folder: str, table_id: str) -> Optional[pd.DataFrame]:
    """
    Load a table from a CSV file.

    Args:
        all_csv_folder (str): Path to the folder containing all CSV files.
        table_id (str): Filename of the table to load.

    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame if successful, else None.
    """
    table_path = os.path.join(all_csv_folder, table_id)
    try:
        table = pd.read_csv(table_path, delimiter="#")
        logging.debug(f"Loaded table {table_id} from {table_path}.")
        return table
    except Exception as e:
        logging.error(f"Failed to load table {table_id} from {table_path}: {e}")
        return None


################################################################################
#                        TABLE FORMAT CONVERTERS
################################################################################

def format_table_to_markdown(table: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to a Markdown-formatted string.
    """
    return table.to_markdown(index=False)

def naturalize_table(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to a naturalized text format that follows the style in TabFact.
    """
    rows = []
    for row_idx, row in df.iterrows():
        row_description = [f"Row {row_idx + 1} is:"]
        for col_name, cell_value in row.items():
            row_description.append(f"{col_name} is {cell_value}")
        rows.append(" ".join(row_description) + ".")
    return " ".join(rows)


################################################################################
#                          PROMPT GENERATION
################################################################################

def generate_prompt(
    table_id: str,
    claim: str,
    all_csv_folder: str,
    learning_type: str = "zero_shot",
    format_type: str = "naturalized",  # "markdown" or "naturalized"
) -> Optional[str]:
    """
    Generate a single prompt for a specific table and claim, 
    instructing the model to produce a JSON with "answer" and "highlighted_cells".
    """
    table = load_table(all_csv_folder, table_id)
    if table is None:
        logging.warning(f"Skipping prompt generation for table {table_id} due to load failure.")
        return None

    if format_type == "markdown":
        table_formatted = format_table_to_markdown(table)
    else:
        table_formatted = naturalize_table(table)

    # Choose the base prompt depending on learning_type & format_type
    if learning_type == "zero_shot":
        if format_type == "markdown":
            prompt = ZERO_SHOT_PROMPT_MARKDOWN.format(
                table_formatted=table_formatted, 
                claim=claim
            )
        else:
            prompt = ZERO_SHOT_PROMPT_NATURALIZED.format(
                table_formatted=table_formatted, 
                claim=claim
            )

    elif learning_type == "one_shot":
        if format_type == "markdown":
            prompt = ONE_SHOT_PROMPT_MARKDOWN.format(
                one_shot_example=ONE_SHOT_EXAMPLE_MARKDOWN,
                table_formatted=table_formatted,
                claim=claim
            )
        else:
            prompt = ONE_SHOT_PROMPT_NATURALIZED.format(
                one_shot_example=ONE_SHOT_EXAMPLE_NATURALIZED,
                table_formatted=table_formatted,
                claim=claim
            )

    elif learning_type == "few_shot":
        if format_type == "markdown":
            prompt = FEW_SHOT_PROMPT_MARKDOWN.format(
                few_shot_example=FEW_SHOT_EXAMPLE_MARKDOWN,
                table_formatted=table_formatted,
                claim=claim
            )
        else:
            prompt = FEW_SHOT_PROMPT_NATURALIZED.format(
                few_shot_example=FEW_SHOT_EXAMPLE_NATURALIZED,
                table_formatted=table_formatted,
                claim=claim
            )
    else:
        logging.error(f"Unsupported learning type: {learning_type}")
        return None

    return prompt.strip()

def extract_json_from_response(raw_response: str) -> dict:
    """
    Attempt to parse JSON from the model response in two ways:
      1. Look for a code fence with ```json ... ```
      2. If none found, try to parse the entire response as JSON.
    Return a dictionary. If nothing works, return {}.
    """
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
    if match:
        # Found a code fence. Parse that snippet
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON decode error (inside code fence): {e}")
            # fallback to entire raw_response parse
            pass

    # If we get here, either no code fence or fence parse failed
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        print("[WARN] Could not parse raw_response as JSON either.")
        # final fallback
        return {}

################################################################################
#                          TEST MODEL FUNCTION
################################################################################

def test_model_on_claims(
    model: OllamaLLM,
    full_cleaned_data: Dict[str, Any],
    test_all: bool = False,
    N: int = 10,
    all_csv_folder: str = "data/all_csv/",
    learning_type: str = "zero_shot",
    format_type: str = "naturalized",
    checkpoint_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Test the model on the first N claims or all claims, using a SINGLE prompt
    that returns JSON of the form:
      {
        "answer": "TRUE" or "FALSE",
        "highlighted_cells": [
          {"row_index": 0, "column_name": "Revenue"},
          ...
        ]
      }

    We'll parse that JSON, store it in the final results, 
    and measure performance as before.
    """
    results = []
    completed_keys = set()

    if checkpoint_file and os.path.exists(checkpoint_file):
        logging.info(f"Loading existing checkpoint from {checkpoint_file}")
        with open(checkpoint_file, "r") as f:
            results = json.load(f)
        for row in results:
            completed_keys.add((row["table_id"], row["claim"]))

    keys = list(full_cleaned_data.keys())
    limit = len(keys) if test_all else min(N, len(keys))
    logging.info(
        f"Testing {limit} tables out of {len(keys)} with checkpoint logic. "
        f"Learning type: {learning_type}, format type: {format_type}"
    )

    for i in tqdm(range(limit), desc="Processing tables"):
        table_id = keys[i]
        claims = full_cleaned_data[table_id][0]
        labels = full_cleaned_data[table_id][1]

        for idx, claim in enumerate(claims):
            if (table_id, claim) in completed_keys:
                continue

            prompt = generate_prompt(
                table_id,
                claim,
                all_csv_folder,
                learning_type,
                format_type
            )
            if prompt is None:
                logging.warning(
                    f"Prompt generation failed (or skipped) for table {table_id}, claim='{claim}'."
                )
                continue

            # -----------------------------
            # Single call to the model
            # -----------------------------
            try:
                raw_response = model.invoke(prompt).strip()
            except Exception as e:
                logging.error(f"Error invoking model for table {table_id}, claim='{claim}': {e}")
                continue

            # -----------------------------
            # Parse the JSON to extract "answer" and "highlighted_cells"
            # If parsing fails, we default to "FALSE" and empty cells
            # -----------------------------
            parsed_answer = "FALSE"
            highlighted_cells = []
            try:
                parsed_dict = extract_json_from_response(raw_response)
                parsed_answer = parsed_dict.get("answer", "FALSE").upper()
                highlighted_cells = parsed_dict.get("highlighted_cells", [])
            except Exception as e:
                logging.warning(f"Failed to parse JSON from model output for table {table_id}, claim='{claim}'. "
                                f"Raw response was:\n{raw_response}")

            predicted_label = 1 if parsed_answer == "TRUE" else 0
            true_label = labels[idx]

            row_result = {
                "table_id": table_id,
                "claim": claim,
                "predicted_response": predicted_label,
                "resp": raw_response,    # store the raw model response for debugging
                "true_response": true_label,
                "highlighted_cells": highlighted_cells  # store the parsed cells
            }

            results.append(row_result)
            completed_keys.add((table_id, claim))

            # Save checkpoint after each claim
            if checkpoint_file:
                with open(checkpoint_file, "w") as f:
                    json.dump(results, f, indent=2)

    return results


# -------------------- PARALLEL CODE (unchanged except if you want to parse JSON) --------------------

def process_claim(model_name: str,
                  table_id: str,
                  claim: str,
                  true_label: int,
                  all_csv_folder: str,
                  learning_type: str,
                  format_type: str) -> Dict[str, Any]:
    """
    Helper function for parallel processing of a single claim.
    (If you want to also parse JSON here, you can replicate the same logic as above.)
    """
    llm = OllamaLLM(model=model_name)

    prompt = generate_prompt(table_id, claim, all_csv_folder, learning_type, format_type)
    if prompt is None:
        return {
            "table_id": table_id,
            "claim": claim,
            "predicted_response": None,
            "resp": "PROMPT_ERROR",
            "true_response": true_label,
            "highlighted_cells": []
        }
    
    raw_response = llm.invoke(prompt).strip()

    # Attempt JSON parse:
    parsed_answer = "FALSE"
    highlighted_cells = []
    try:
        parsed_dict = extract_json_from_response(raw_response)
        parsed_answer = parsed_dict.get("answer", "FALSE").upper()
        highlighted_cells = parsed_dict.get("highlighted_cells", [])
    except:
        pass

    predicted_label = 1 if parsed_answer == "TRUE" else 0

    return {
        "table_id": table_id,
        "claim": claim,
        "predicted_response": predicted_label,
        "resp": raw_response,
        "true_response": true_label,
        "highlighted_cells": highlighted_cells
    }


def test_model_on_claims_parallel(
    model_name: str,
    full_cleaned_data: Dict[str, Any],
    test_all: bool = False,
    N: int = 10,
    all_csv_folder: str = "data/all_csv/",
    learning_type: str = "zero_shot",
    format_type: str = "naturalized",
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
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
                    process_claim,
                    model_name,
                    table_id,
                    claim,
                    labels[idx],
                    all_csv_folder,
                    learning_type,
                    format_type
                ))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing claims"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Error in parallel claim processing: {e}")
                continue

    return results


def process_model(
    model_name: str,
    learning_types: List[str],
    datasets: List[Dict[str, Any]],
    repo_folder: str,
    all_csv_folder: str,
    test_all: bool,
    N: int,
    batch_prompts: bool = False,
    max_workers: int = 4,
    checkpoint_folder: str = "checkpoints",
    format_type: str = "naturalized",
    results_folder: str = "results",
):
    logging.info(f"[process_model] Initializing model: {model_name}")

    for learning_type in learning_types:
        for dataset in datasets:
            dataset_type, dataset_data = next(iter(dataset.items()))
            logging.info(
                f"[process_model] Model={model_name}, "
                f"Dataset={dataset_type}, LearningType={learning_type}"
            )
            try:
                if batch_prompts:
                    results = test_model_on_claims_parallel(
                        model_name=model_name,
                        full_cleaned_data=dataset_data,
                        test_all=test_all,
                        N=N,
                        all_csv_folder=os.path.join(repo_folder, all_csv_folder),
                        learning_type=learning_type,
                        format_type=format_type,
                        max_workers=max_workers
                    )
                else:
                    try:
                        llm = OllamaLLM(model=model_name)
                    except Exception as e:
                        logging.error(f"Failed to initialize model {model_name}: {e}")
                        continue

                    checkpoint_file = f"{checkpoint_folder}/checkpoint_{model_name}_{dataset_type}_{'all' if test_all else N}_{learning_type}_{format_type}.json"
                    os.makedirs(checkpoint_folder, exist_ok=True)
                    
                    results = test_model_on_claims(
                        model=llm,
                        full_cleaned_data=dataset_data,
                        test_all=test_all,
                        N=N,
                        all_csv_folder=os.path.join(repo_folder, all_csv_folder),
                        learning_type=learning_type,
                        format_type=format_type,
                        checkpoint_file=checkpoint_file
                    )

                saving_directory = f"{results_folder}/results_plots_{model_name}_{dataset_type}_{'all' if test_all else N}_{learning_type}_{format_type}"
                os.makedirs(saving_directory, exist_ok=True)

                calculate_and_plot_metrics(
                    results=results,
                    save_dir=saving_directory,
                    save_stats_file="summary_stats.json",
                    learning_type=learning_type,
                    dataset_type=dataset_type,
                    model_name=model_name,
                    format_type=format_type
                )

                with open(f"{results_folder}/results_with_cells_{model_name}_{dataset_type}_{'all' if test_all else N}_{learning_type}_{format_type}.json", "w") as f:
                    json.dump(results, f, indent=2)

                logging.info(f"Wrote results to {results_folder}/results_with_cells_{model_name}_{dataset_type}_{'all' if test_all else N}_{learning_type}_{format_type}.json")
                
            except Exception as e:
                logging.error(f"Error in process_model for {model_name}: {e}")
                continue

################################################################################
#                            PLOTTING FUNCTIONS
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
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
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
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (area = {roc_auc:.2f})",
    )
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


################################################################################
#                       METRICS + STATS FUNCTIONS
################################################################################

def calculate_and_plot_metrics(
    results: List[Dict[str, Any]],
    save_dir: str = "results_plots",
    save_stats_file: str = "summary_stats.pkl",
    learning_type: str = "",
    dataset_type: str = "",
    model_name: str = "",
    format_type: str = "naturalized",
) -> None:
    y_true = [result["true_response"] for result in results]
    y_pred = [result["predicted_response"] for result in results]
    classes = [0, 1]

    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Saving results to directory: {save_dir}")

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0, 0] if cm.shape[0] > 0 else 0
        fp = cm[0, 1] if cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        tp = cm[1, 1] if cm.shape[1] > 1 and cm.shape[0] > 1 else 0

    cm_title = f"Confusion Matrix - {learning_type.capitalize()} Learning - {dataset_type.replace('_', ' ').capitalize()} Dataset"
    cm_save_path = os.path.join(save_dir, f"confusion_matrix_{learning_type}_{dataset_type}.png")
    plot_confusion_matrix_plot(y_true, y_pred, classes, cm_title, cm_save_path)

    roc_title = f"ROC Curve - {learning_type.capitalize()} Learning - {dataset_type.replace('_', ' ').capitalize()} Dataset"
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
        logging.error(f"Failed to save summary statistics to {stats_save_path}: {e}")

    logging.info(f"{learning_type.capitalize()} Learning - {dataset_type.replace('_', ' ').capitalize()} Dataset:")
    logging.info(f"Precision: {precision:.2f}")
    logging.info(f"Recall: {recall:.2f}")
    logging.info(f"F1 Score: {f1:.2f}")
    logging.info(f"Accuracy: {accuracy:.2f}")
    logging.info(f"True Positives (TP): {tp}")
    logging.info(f"False Positives (FP): {fp}")
    logging.info(f"True Negatives (TN): {tn}")
    logging.info(f"False Negatives (FN): {fn}")


################################################################################
#                          JSON LOADING FUNCTION
################################################################################

def load_json_file(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logging.info(f"Loaded data from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"Failed to load JSON file {file_path}: {e}")
        return {}


################################################################################
#                                MAIN FUNCTION
################################################################################

def main(batch_prompts=False, parallel_models=False, max_workers=4) -> None:
    repo_folder = "../original_repo"

    file_simple_r1 = "collected_data/r1_training_all.json" 
    file_complex_r2 = "collected_data/r2_training_all.json" 
    all_csv_folder = "data/all_csv/"  
    test_set = "tokenized_data/test_examples.json"
    val_set = "tokenized_data/val_examples.json"

    checkpoint_folder = "checkpoints"
    results_folder = f"results_{datetime.now().strftime('%Y%m%d')}"

    # Merge example if needed
    # output_file = "full_claim_file.json"
    # merge_training_data(file_simple_r1, file_complex_r2, output_file)

    # Load data
    r1_simple = load_json_file(os.path.join(repo_folder, file_simple_r1))
    r2_complex = load_json_file(os.path.join(repo_folder, file_complex_r2))
    test_json = load_json_file(os.path.join(repo_folder, test_set))
    val_json = load_json_file(os.path.join(repo_folder, val_set))

    logging.info(f"Current directory: {os.getcwd()}")

    test_all = False
    N = 2

    models = ["mistral"] #, "llama3.2"] #, "phi4"]
    learning_types = ["zero_shot"] #, "one_shot", "few_shot"] 
    datasets = [{"test_set": test_json}, {"val_set": val_json}]
    format_type = "naturalized" 

    if not parallel_models:
        for model_name in models:
            logging.info(f"Initializing model: {model_name}")
            try:
                llm = OllamaLLM(model=model_name)
            except Exception as e:
                logging.error(f"Failed to initialize model {model_name}: {e}")
                continue

            for learning_type in learning_types:
                for dataset in datasets:
                    dataset_type, dataset_data = next(iter(dataset.items()))
                    logging.info(f"Processing dataset: {dataset_type} with learning type: {learning_type}")

                    checkpoint_file = f"{checkpoint_folder}/checkpoint_{model_name}_{dataset_type}_{'all' if test_all else N}_{learning_type}_{format_type}.json"
                    os.makedirs(checkpoint_folder, exist_ok=True)

                    if not batch_prompts:
                        # SINGLE-PROMPT approach with JSON parse
                        results = test_model_on_claims(
                            model=llm,
                            full_cleaned_data=dataset_data,
                            test_all=test_all,
                            N=N,
                            all_csv_folder=os.path.join(repo_folder, all_csv_folder),
                            learning_type=learning_type,
                            format_type=format_type,
                            checkpoint_file=checkpoint_file
                        )
                    else:
                        results = test_model_on_claims_parallel(
                            model_name=model_name,
                            full_cleaned_data=dataset_data,
                            test_all=test_all,
                            N=N,
                            all_csv_folder=os.path.join(repo_folder, all_csv_folder),
                            learning_type=learning_type,
                            format_type=format_type,
                            max_workers=max_workers
                        )

                    saving_directory = f"{results_folder}/results_plots_{model_name}_{dataset_type}_{'all' if test_all else N}_{learning_type}_{format_type}"
                    os.makedirs(saving_directory, exist_ok=True)

                    calculate_and_plot_metrics(
                        results=results,
                        save_dir=saving_directory,
                        save_stats_file="summary_stats.json",
                        learning_type=learning_type,
                        dataset_type=dataset_type,
                        model_name=model_name,
                        format_type=format_type
                    )

                    with open(f"{results_folder}/results_with_cells_{model_name}_{dataset_type}_{'all' if test_all else N}_{learning_type}_{format_type}.json", "w") as f:
                        json.dump(results, f, indent=2)

                    logging.info(f"Wrote results to {results_folder}/results_with_cells_{model_name}_{dataset_type}_{'all' if test_all else N}_{learning_type}_{format_type}.json")
    else:
        tasks = []
        with ProcessPoolExecutor(max_workers=len(models)) as executor:
            for model_name in models:
                tasks.append(executor.submit(
                    process_model,
                    model_name,
                    learning_types,
                    datasets,
                    repo_folder,
                    all_csv_folder,
                    test_all,
                    N,
                    batch_prompts,
                    max_workers,
                    checkpoint_folder,
                    format_type,
                    results_folder
                ))

            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error while processing model in parallel: {e}")


################################################################################
#                         LOAD AND PRINT STATS
################################################################################

def load_and_print_stats(json_file_path: str) -> None:
    if not os.path.exists(json_file_path):
        print(f"Error: The file {json_file_path} does not exist.")
        return

    try:
        with open(json_file_path, "r") as f:
            stats = json.load(f)
        print("Summary Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error loading or reading the JSON file: {e}")


################################################################################
#                              ENTRY POINT
################################################################################

if __name__ == "__main__":
    # gc.collect()
    # torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Load and print stats from a json file or run the script as normal.")
    parser.add_argument(
        "-s", 
        "--stats_file",
        type=str,
        help="Path to the json file containing summary statistics."
    )
    parser.add_argument(
        "-p",
        "--parallel_models",
        action="store_true",
        help="Run the script in parallel mode for multiple models."
    )
    parser.add_argument(
        "-b",
        "--batch_prompts",
        action="store_true",
        help="Run the script in batch mode for multiple prompts. NOTE: In this mode, checkpointing is disabled."
    )
    parser.add_argument("--max_workers", type=int, default=4,
        help="Number of worker processes to use for parallel/batch runs."
    )

    args = parser.parse_args()

    if args.stats_file:
        load_and_print_stats(args.stats_file)
    else:
        main(
            batch_prompts=args.batch_prompts,
            parallel_models=args.parallel_models,
            max_workers=args.max_workers
        )