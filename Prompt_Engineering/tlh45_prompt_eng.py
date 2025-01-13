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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

################################################################################
#                           PROMPT TEMPLATES
################################################################################

END_OF_PROMPT_INSTRUCTIONS = """
Your answer should be ONLY "TRUE" or "FALSE" (all caps), with no additional explanation.
"""

# -------------------------- ZERO-SHOT PROMPTS --------------------------
ZERO_SHOT_PROMPT_MARKDOWN = """
You are tasked with determining whether a claim about the following table (in Markdown) is TRUE or FALSE.

Table (Markdown):
{table_formatted}

Claim: "{claim}"

Instructions:
- Carefully check each condition in the claim against the table.
- Respond with "TRUE" if the claim is fully supported by the table data.
- Respond with "FALSE" if it is not supported.

""" + END_OF_PROMPT_INSTRUCTIONS

ZERO_SHOT_PROMPT_NATURALIZED = """
You are tasked with determining whether a claim about the following table (in natural text) is TRUE or FALSE.

Table (Naturalized):
{table_formatted}

Claim: "{claim}"

Instructions:
- Verify if the claim is supported by the data described in each row.
- If yes, respond "TRUE".
- Otherwise, respond "FALSE".

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
- The claim says B is cheaper, but B is actually more expensive.
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
- Compare the claim to the table data.
- Respond "TRUE" if the claim is fully supported, else "FALSE".

""" + END_OF_PROMPT_INSTRUCTIONS

ONE_SHOT_EXAMPLE_NATURALIZED = """
Example of evaluating a claim with a table in naturalized text:

Row 1: Product is A, Price is 10, Stock is 100.
Row 2: Product is B, Price is 15, Stock is 50.

Claim: "Product A is the most expensive item."

Analysis and Answer:
- A costs 10, B costs 15.
- B is more expensive than A.
- Therefore, the claim is FALSE.
"""

ONE_SHOT_PROMPT_NATURALIZED = """
Below is an example of how to evaluate a claim using a table in naturalized text:

{one_shot_example}

Now, evaluate the following table and claim in the same manner.

Table (Naturalized):
{table_formatted}

Claim: "{claim}"

Instructions:
- Check if the claim is fully supported by the data or not.
- Answer "TRUE" or "FALSE" accordingly.

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
- Check if the data supports the claim. If yes, respond "TRUE"; otherwise "FALSE".

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

    Args:
        table (pd.DataFrame): The DataFrame to convert.

    Returns:
        str: Markdown-formatted table.
    """
    return table.to_markdown(index=False)

def naturalize_table(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to a naturalized text format that follows the style in TabFact.

    Args:
        df (pd.DataFrame): The DataFrame to convert.

    Returns:
        str: Naturalized text description of the table.
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
    Generate a prompt for a specific table and claim.

    Args:
        table_id (str): The table filename from full_cleaned.json.
        claim (str): The statement (claim) to be validated.
        all_csv_folder (str): Path to the folder containing CSV tables.
        learning_type (str, optional): Type of prompt engineering ("zero_shot", "one_shot", "few_shot").
        format_type (str, optional): "markdown" or "naturalized". Defaults to "naturalized".

    Returns:
        Optional[str]: A formatted prompt string or None if table loading fails.
    """
    table = load_table(all_csv_folder, table_id)
    if table is None:
        logging.warning(f"Skipping prompt generation for table {table_id} due to load failure.")
        return None

    # Convert table
    if format_type == "markdown":
        table_formatted = format_table_to_markdown(table)
    elif format_type == "naturalized":
        table_formatted = naturalize_table(table)
    else:
        logging.error(f"Unsupported format type: {format_type}")
        return None

    # Pick the correct prompt template based on learning_type + format_type
    if learning_type == "zero_shot":
        if format_type == "markdown":
            prompt = ZERO_SHOT_PROMPT_MARKDOWN.format(table_formatted=table_formatted, claim=claim)
        else:
            prompt = ZERO_SHOT_PROMPT_NATURALIZED.format(table_formatted=table_formatted, claim=claim)

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

    logging.debug(f"Generated prompt for table {table_id} and claim '{claim}'.")
    return prompt.strip()


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
    Test the model on the first N claims or all claims from full_cleaned.json.
    With checkpoint logic to resume partial progress.
    """
    # ---------------------
    # 1) Load existing checkpoint data if any
    # ---------------------
    results = []        # We'll store old + new results here
    completed_keys = set()

    if checkpoint_file and os.path.exists(checkpoint_file):
        logging.info(f"Loading existing checkpoint from {checkpoint_file}")
        with open(checkpoint_file, "r") as f:
            # Existing results are loaded directly into 'results'
            results = json.load(f)

        # Build a set of (table_id, claim) that are already done
        for row in results:
            completed_keys.add((row["table_id"], row["claim"]))

    # Prepare the list of keys/tables
    keys = list(full_cleaned_data.keys())
    limit = len(keys) if test_all else min(N, len(keys))
    logging.info(
        f"Testing {limit} tables out of {len(keys)} with checkpoint logic. "
        f"Learning type: {learning_type}, format type: {format_type}"
    )

    # ---------------------
    # 2) Main inference loop
    # ---------------------
    for i in tqdm(range(limit), desc="Processing tables"):
        table_id = keys[i]
        claims = full_cleaned_data[table_id][0]
        labels = full_cleaned_data[table_id][1]

        for idx, claim in enumerate(claims):
            # Skip if already in checkpoint
            if (table_id, claim) in completed_keys:
                continue

            # Generate the prompt
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

            # ---------------------
            # 3) Actually run inference
            # ---------------------
            try:
                response = model.invoke(prompt).strip()
                predicted_label = 1 if "TRUE" in response.upper() else 0
                true_label = labels[idx]

                row_result = {
                    "table_id": table_id,
                    "claim": claim,
                    "predicted_response": predicted_label,
                    "resp": response,
                    "true_response": true_label,
                }

                # Add new result to 'results' and mark completed
                results.append(row_result)
                completed_keys.add((table_id, claim))

                # ---------------------
                # 4) Save checkpoint after each new claim (or batch it up)
                # ---------------------
                if checkpoint_file:
                    with open(checkpoint_file, "w") as f:
                        json.dump(results, f, indent=2)

            except Exception as e:
                logging.error(f"Error invoking model for table {table_id}, claim='{claim}': {e}")
                continue

    # Return the full list of results (old + new)
    return results


# multiproeccesing OPTIONS
def process_claim(model_name: str,
                  table_id: str,
                  claim: str,
                  true_label: int,
                  all_csv_folder: str,
                  learning_type: str,
                  format_type: str) -> Dict[str, Any]:
    """
    Helper function for parallel processing of a single claim.
    """
    # We need a local instance of OllamaLLM if the model is not picklable across processes
    # or if Ollama can handle parallel calls from one global instance. 
    # Typically, you'd re-initialize the model in each worker if needed.
    llm = OllamaLLM(model=model_name)

    prompt = generate_prompt(table_id, claim, all_csv_folder, learning_type, format_type)
    if prompt is None:
        # Return something that indicates we skipped
        return {
            "table_id": table_id,
            "claim": claim,
            "predicted_response": None,
            "resp": "PROMPT_ERROR",
            "true_response": true_label
        }
    
    response = llm.invoke(prompt).strip()
    predicted_label = 1 if "TRUE" in response.upper() else 0
    return {
        "table_id": table_id,
        "claim": claim,
        "predicted_response": predicted_label,
        "resp": response,
        "true_response": true_label,
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
    """
    Test the model on the first N claims (or all) from full_cleaned.json in parallel using ProcessPoolExecutor.
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
    """
    Runs the entire pipeline (datasets, claims) for a single model.
    """
    logging.info(f"[process_model] Initializing model: {model_name}")

    # We might STILL do a local instantiation here, if you want to do test_model_on_claims( ) single-thread, 
    # but you can skip if you are only doing parallel claims and re-initialize in each claim.
    #
    # If you prefer a single-thread approach for claims, you might keep a single instance of the model:
    # try:
    #     llm = OllamaLLM(model=model_name)
    # except Exception as e:
    #     logging.error(f"Failed to initialize model {model_name} at process_model level: {e}")
    #     return

    for learning_type in learning_types:
        for dataset in datasets:
            dataset_type, dataset_data = next(iter(dataset.items()))
            logging.info(
                f"[process_model] Model={model_name}, "
                f"Dataset={dataset_type}, LearningType={learning_type}"
            )
            try:
                # Call your concurrency-based approach for claims:
                if batch_prompts:
                    results = test_model_on_claims_parallel(
                        model_name=model_name,
                        full_cleaned_data=dataset_data,
                        test_all=test_all,
                        N=N,
                        all_csv_folder=os.path.join(repo_folder, all_csv_folder),
                        learning_type=learning_type,
                        format_type=format_type,
                        max_workers=max_workers  # or whatever you want
                    )
                else:
                    try:
                        llm = OllamaLLM(model=model_name)
                    except Exception as e:
                        logging.error(f"Failed to initialize model {model_name}: {e}")
                        continue

                    checkpoint_file = f"{checkpoint_folder}/checkpoint_{model_name}_{dataset_type}_{'all' if test_all else N}_{learning_type}_{format_type}.json"
                    results = test_model_on_claims(
                        model=llm,
                        full_cleaned_data=dataset_data,
                        test_all=test_all,
                        N=N,
                        all_csv_folder=os.path.join(repo_folder, all_csv_folder),
                        learning_type=learning_type,
                        format_type=format_type,  # or "markdown"
                        checkpoint_file=checkpoint_file
                    )

                # Save & plot
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
    """
    Plot and save the confusion matrix.

    Args:
        y_true (List[int]): True labels.
        y_pred (List[int]): Predicted labels.
        classes (List[int]): List of class labels.
        title (str): Title of the plot.
        save_path (str): File path to save the plot.
    """
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
    """
    Plot and save the ROC curve.

    Args:
        y_true (List[int]): True labels.
        y_pred (List[int]): Predicted labels.
        title (str): Title of the plot.
        save_path (str): File path to save the plot.
    """
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
    """
    Calculate precision, recall, F1-score, accuracy, TP, FP, TN, FN.
    Plot confusion matrix and ROC curve,
    Save all to files.

    Args:
        results (List[Dict[str, Any]]): List of dictionaries containing "true_response" and "predicted_response".
        save_dir (str, optional): Directory to save plots and stats. Defaults to "results_plots".
        save_stats_file (str, optional): Filename to save stats in pickle format. Defaults to "summary_stats.pkl".
        learning_type (str, optional): Type of learning method ("zero_shot", "one_shot", "few_shot"). Defaults to "".
        dataset_type (str, optional): Type of dataset ("test_set", "val_set"). Defaults to "".
    """
    y_true = [result["true_response"] for result in results]
    y_pred = [result["predicted_response"] for result in results]
    classes = [0, 1]

    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Saving results to directory: {save_dir}")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0, 0] if cm.shape[0] > 0 else 0
        fp = cm[0, 1] if cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        tp = cm[1, 1] if cm.shape[1] > 1 and cm.shape[0] > 1 else 0

    # Plot and save confusion matrix
    cm_title = f"Confusion Matrix - {learning_type.capitalize()} Learning - {dataset_type.replace('_', ' ').capitalize()} Dataset"
    cm_save_path = os.path.join(save_dir, f"confusion_matrix_{learning_type}_{dataset_type}.png")
    plot_confusion_matrix_plot(y_true, y_pred, classes, cm_title, cm_save_path)

    # Plot and save ROC curve
    roc_title = f"ROC Curve - {learning_type.capitalize()} Learning - {dataset_type.replace('_', ' ').capitalize()} Dataset"
    roc_save_path = os.path.join(save_dir, f"roc_curve_{learning_type}_{dataset_type}.png")
    plot_roc_curve_plot(y_true, y_pred, roc_title, roc_save_path)

    # Calculate evaluation metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Compile statistics
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
        """
        Recursively convert non-JSON-serializable objects like numpy types to native Python types.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to lists
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        else:
            return obj  # Return as-is for already serializable objects

    # Save statistics to JSON file
    stats_save_path = os.path.join(save_dir, save_stats_file)
    try:
        serializable_stats = convert_to_serializable(stats)
        with open(stats_save_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)
        logging.info(f"Saved summary statistics to {stats_save_path}.")
    except Exception as e:
        logging.error(f"Failed to save summary statistics to {stats_save_path}: {e}")

    # Log metrics
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
    """
    Load a JSON file and return its content.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON data.
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
#                                MAIN FUNCTION
################################################################################

def main(batch_prompts=False, parallel_models=False, max_workers=4) -> None:
    """
    Main execution function.
    """
    # Initial data paths
    repo_folder = "../original_repo"

    file_simple_r1 = "collected_data/r1_training_all.json"  # File with simple claims
    file_complex_r2 = "collected_data/r2_training_all.json"  # File with complex claims
    all_csv_folder = "data/all_csv/"  # Folder with the tables themselves
    test_set = "tokenized_data/test_examples.json"
    val_set = "tokenized_data/val_examples.json"

    checkpoint_folder = "checkpoints"
    results_folder = f"results_{datetime.now().strftime('%Y%m%d')}"

    # Merge example (commented out if not needed)
    # output_file = "full_claim_file.json"
    # merge_training_data(file_simple_r1, file_complex_r2, output_file)
    # with open(output_file, "r") as f:
    #     claim_file = json.load(f) # file containing the claims for each table

    # Load data
    r1_simple = load_json_file(os.path.join(repo_folder, file_simple_r1))
    r2_complex = load_json_file(os.path.join(repo_folder, file_complex_r2))
    test_json = load_json_file(os.path.join(repo_folder, test_set))
    val_json = load_json_file(os.path.join(repo_folder, val_set))

    logging.info(f"Current directory: {os.getcwd()}")

    # Parameters
    test_all = True
    N = 3  # Number of claims to test (if test_all is False)

    summary_metrics = {}
    models = ["mistral", "llama3.2", "phi4"]  # https://ollama.com/library
    learning_types = ["zero_shot", "one_shot", "few_shot"]  # Example learning types

    # Define datasets (test_set, val_set) – or you could also do r1_simple/r2_complex
    datasets = [{"test_set": test_json}, {"val_set": val_json}]
    format_type = "naturalized"  # or "markdown"

    # For example, if you want to try simple or complex sets, you could do:
    # datasets = [{"simple_set": r1_simple}, {"complex_set": r2_complex}]

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

                    if not batch_prompts:
                        results = test_model_on_claims(
                            model=llm,
                            full_cleaned_data=dataset_data,
                            test_all=test_all,
                            N=N,
                            all_csv_folder=os.path.join(repo_folder, all_csv_folder),
                            learning_type=learning_type,
                            format_type=format_type,  # or "markdown"
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
                            max_workers=max_workers  # or whatever you want
                        )

                    saving_directory = f"{results_folder}/results_plots_{model_name}_{dataset_type}_{'all' if test_all else N}_{learning_type}_{format_type}"
                    os.makedirs(saving_directory, exist_ok=True)

                    # Save accuracies, plots in appropriate folder
                    calculate_and_plot_metrics(
                        results=results,
                        save_dir=saving_directory,
                        save_stats_file="summary_stats.json",
                        learning_type=learning_type,
                        dataset_type=dataset_type,
                        model_name=model_name,
                        format_type=format_type
                    )
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
                    future.result()  # we don't return anything, just gather
                except Exception as e:
                    logging.error(f"Error while processing model in parallel: {e}")


################################################################################
#                         LOAD AND PRINT STATS
################################################################################

def load_and_print_stats(json_file_path: str) -> None:
    """
    Load the summary statistics from a pickle file and print them.

    Args:
        pickle_file_path (str): Path to the pickle file containing summary statistics.
    """
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

    # run garbage collection & clean cuda cache
    # gc.collect()
    # torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Load and print stats from a pickle file or run the script as normal.")
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
        help="Run the script in batch mode for multiple prompts."
    )
    parser.add_argument("--max_workers", type=int, default=4,
        help="Number of worker processes to use for parallel/batch runs."
    )

    args = parser.parse_args()

    if args.stats_file:
        load_and_print_stats(args.stats_file)
    else:
        main(
            batch_prompts=args.batch_prompts
            , parallel_models=args.parallel_models
            , max_workers=args.max_workers
        )