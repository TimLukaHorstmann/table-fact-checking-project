#!/usr/bin/env python3
"""
prompt_engineering.py

This script implements the prompt engineering approach by subclassing the BaseFactChecker.
It defines the prompt templates and the generate_prompt() method and provides a main function
to run experiments. It supports two levels of parallelization:
  1. Batch processing (parallel execution of claims) via --batch_prompts and --max_workers.
  2. Parallel processing of multiple models via --parallel_models and --models.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from factchecker_base import BaseFactChecker, test_model_on_claims, test_model_on_claims_parallel, calculate_and_plot_metrics, load_json_file
from langchain_ollama import OllamaLLM
from typing import Optional

# Define prompt templates specific to prompt engineering.
END_OF_PROMPT_INSTRUCTIONS = """
Return only a valid JSON object with two keys:
"answer": must be "TRUE" or "FALSE" (all caps)
"relevant_cells": a list of objects, each with "row_index" (int) and "column_name" (string)

For example:

{
  "answer": "TRUE",
  "relevant_cells": [
    {"row_index": 0, "column_name": "Revenue"},
    {"row_index": 1, "column_name": "Employees"}
  ]
}

No extra keys, no extra text. Just that JSON.
"""

ZERO_SHOT_PROMPT_MARKDOWN = """
You are tasked with determining whether a claim about the following table (in Markdown) is TRUE or FALSE.

Table (Markdown):
{table_formatted}

Claim: "{claim}"

Instructions:
- Carefully check each condition in the claim against the table and determine which cells are relevant.
- If fully supported, answer "TRUE"; otherwise, answer "FALSE".
""" + END_OF_PROMPT_INSTRUCTIONS

ZERO_SHOT_PROMPT_NATURALIZED = """
You are tasked with determining whether a claim about the following table (in natural text) is TRUE or FALSE.

Table (Naturalized):
{table_formatted}

Claim: "{claim}"

Instructions:
- If the claim is supported by the data, answer "TRUE"; otherwise, answer "FALSE".
""" + END_OF_PROMPT_INSTRUCTIONS

ONE_SHOT_EXAMPLE_MARKDOWN = """
Example:
| Product | Price | Stock |
|---------|-------|-------|
| A       | 10    | 100   |
| B       | 15    | 50    |

Claim: "Product B is cheaper than Product A."
Answer: FALSE.
"""

ONE_SHOT_PROMPT_MARKDOWN = """
Below is an example of how to evaluate a claim using a Markdown table:

{one_shot_example}

Now, evaluate the following table and claim.

Table (Markdown):
{table_formatted}

Claim: "{claim}"
""" + END_OF_PROMPT_INSTRUCTIONS

ONE_SHOT_EXAMPLE_NATURALIZED = """
Example:
Row 1: Product is A, Price is 10, Stock is 100.
Row 2: Product is B, Price is 15, Stock is 50.

Claim: "Product A is the most expensive item."
Answer: FALSE.
"""

ONE_SHOT_PROMPT_NATURALIZED = """
Below is an example of how to evaluate a claim using naturalized text:

{one_shot_example}

Now, evaluate the following table and claim.

Table (Naturalized):
{table_formatted}

Claim: "{claim}"
""" + END_OF_PROMPT_INSTRUCTIONS

FEW_SHOT_EXAMPLE_MARKDOWN = """
Example 1:
Table (Markdown):
| Country | Population |
|---------|-----------|
| X       | 10        |
| Y       | 20        |

Claim: "Country X has a larger population than Country Y."
Answer: FALSE.

Example 2:
Table (Markdown):
| Company | Revenue | Employees |
|---------|---------|-----------|
| A       | 100     | 500       |
| B       | 200     | 1000      |

Claim: "Company A has fewer employees than Company B."
Answer: TRUE.
"""

FEW_SHOT_PROMPT_MARKDOWN = """
You will see multiple examples of how to evaluate a claim using Markdown tables:

{few_shot_example}

Now, evaluate a new table and claim.

Table (Markdown):
{table_formatted}

Claim: "{claim}"
""" + END_OF_PROMPT_INSTRUCTIONS

FEW_SHOT_EXAMPLE_NATURALIZED = """
Example 1:
Row 1: Country is X, Population is 10.
Row 2: Country is Y, Population is 20.

Claim: "Country X has a larger population than Country Y."
Answer: FALSE.

Example 2:
Row 1: Company is A, Revenue is 100, Employees is 500.
Row 2: Company is B, Revenue is 200, Employees is 1000.

Claim: "Company B has more revenue than Company A."
Answer: TRUE.
"""

FEW_SHOT_PROMPT_NATURALIZED = """
You will see multiple examples of how to evaluate a claim using naturalized text:

{few_shot_example}

Now, evaluate a new table and claim.

Table (Naturalized):
{table_formatted}

Claim: "{claim}"
""" + END_OF_PROMPT_INSTRUCTIONS

################################################################################
#                 PROMPT ENGINEERING FACT CHECKER CLASS
################################################################################

class PromptEngineeringFactChecker(BaseFactChecker):
    def generate_prompt(self, table_id: str, claim: str) -> Optional[str]:
        table = self.load_table(table_id)
        if table is None:
            logging.warning(f"Table {table_id} could not be loaded.")
            return None
        table_formatted = self.format_table(table)
        if self.learning_type == "zero_shot":
            if self.format_type == "markdown":
                prompt = ZERO_SHOT_PROMPT_MARKDOWN.format(
                    table_formatted=table_formatted,
                    claim=claim
                )
            else:
                prompt = ZERO_SHOT_PROMPT_NATURALIZED.format(
                    table_formatted=table_formatted,
                    claim=claim
                )
        elif self.learning_type == "one_shot":
            if self.format_type == "markdown":
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
        elif self.learning_type == "few_shot":
            if self.format_type == "markdown":
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
            logging.error(f"Unsupported learning type: {self.learning_type}")
            return None
        return prompt.strip()

################################################################################
#                         PIPELINE FUNCTION
################################################################################

def run_pipeline_for_model(model_name: str, args):
    """
    Run the entire prompt engineering pipeline for one model.
    Depending on args.batch_prompts, claims are processed serially or in parallel.
    """
    repo_folder = args.repo_folder
    csv_folder = os.path.join(repo_folder, args.csv_folder)
    dataset_file = os.path.join(repo_folder, args.dataset)
    dataset_data = load_json_file(dataset_file)
    
    # Initialize LLM model.
    try:
        if "deepseek" in model_name:
            model = OllamaLLM(model=model_name, params={"n_ctx": 4096, "n_batch": 256})
        else:
            model = OllamaLLM(model=model_name)
    except Exception as e:
        logging.error(f"Failed to initialize model {model_name}: {e}")
        return

    # Create instance of the fact checker.
    fact_checker = PromptEngineeringFactChecker(
        all_csv_folder=csv_folder,
        learning_type=args.learning_type,
        format_type=args.format_type,
        model=model
    )
    
    # Process claims serially or in batch (parallel) mode.
    if args.batch_prompts:
        results = test_model_on_claims_parallel(
            fact_checker_class=PromptEngineeringFactChecker,
            model_name=model_name,
            full_cleaned_data=dataset_data,
            all_csv_folder=csv_folder,
            learning_type=args.learning_type,
            format_type=args.format_type,
            approach="prompt_engineering",
            test_all=False,
            N=args.N,
            max_workers=args.max_workers
        )
    else:
        results = test_model_on_claims(
            fact_checker=fact_checker,
            full_cleaned_data=dataset_data,
            test_all=False,
            N=args.N,
            checkpoint_file=None
        )
    
    # Save results and compute metrics.
    results_folder = f"results_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(results_folder, f"results_prompt_engineering_{model_name}_{args.learning_type}_{args.format_type}_{args.N}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_file}")
    
    metrics_folder = os.path.join(results_folder, f"plots_prompt_engineering_{model_name}_{args.learning_type}_{args.format_type}_{args.N}")
    os.makedirs(metrics_folder, exist_ok=True)
    calculate_and_plot_metrics(
        results=results,
        save_dir=metrics_folder,
        learning_type=args.learning_type,
        dataset_type="test_set",
        model_name=model_name,
        format_type=args.format_type
    )

################################################################################
#                              MAIN FUNCTION
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Run Prompt Engineering Fact Checking Approach")
    parser.add_argument("--repo_folder", type=str, default="../original_repo", help="Path to repository folder")
    parser.add_argument("--csv_folder", type=str, default="data/all_csv/", help="Folder containing CSV tables")
    parser.add_argument("--dataset", type=str, default="tokenized_data/test_examples.json", help="Path to dataset JSON file")
    parser.add_argument("--models", type=str, default="mistral", help="Comma-separated list of model names")
    parser.add_argument("--parallel_models", action="store_true", help="Run different models in parallel")
    parser.add_argument("--batch_prompts", action="store_true", help="Process claims in parallel for each model")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of worker processes for parallel claims")
    parser.add_argument("--learning_type", type=str, default="zero_shot", choices=["zero_shot", "one_shot", "few_shot"])
    parser.add_argument("--format_type", type=str, default="naturalized", choices=["markdown", "naturalized"])
    parser.add_argument("--N", type=int, default=5, help="Number of tables to test")
    args = parser.parse_args()
    
    model_list = [m.strip() for m in args.models.split(",")]
    
    if args.parallel_models:
        with ProcessPoolExecutor(max_workers=len(model_list)) as executor:
            futures = []
            for model_name in model_list:
                futures.append(executor.submit(run_pipeline_for_model, model_name, args))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error in parallel model execution: {e}")
    else:
        for model_name in model_list:
            run_pipeline_for_model(model_name, args)

if __name__ == "__main__":
    main()
