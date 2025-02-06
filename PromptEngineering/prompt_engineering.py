#!/usr/bin/env python3
"""
prompt_engineering.py

This script implements the prompt engineering approach by subclassing the BaseFactChecker.
It defines the prompt templates and the generate_prompt() method and provides a main function
to run experiments. It supports two levels of parallelization:
  1. Batch processing (parallel execution of claims) via --batch_prompts and --max_workers.
  2. Parallel processing of multiple models and combinations via --parallel_models and comma‐separated inputs.
  
In addition, the user may pass multiple datasets, learning types, and format types (as comma‐separated
lists) and the code will run over all combinations.
"""

"""
EXECUTE AS FOLLOWS:

python prompt_engineering.py \
  --repo_folder ../original_repo \
  --csv_folder data/all_csv/ \
  --dataset tokenized_data/test_examples.json,tokenized_data/val_examples.json \
  --learning_type zero_shot,one_shot \
  --format_type markdown,naturalized \
  --models mistral:latest,llama3.2:latest,phi4:latest \
  --parallel_models \
  --batch_prompts \
  --max_workers 4 \
  --N 2

"""



import os
import json
import logging
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from typing import Optional

from factchecker_base import BaseFactChecker, test_model_on_claims, test_model_on_claims_parallel, calculate_and_plot_metrics, load_json_file
from langchain_ollama import OllamaLLM

# Updated prompt templates with literal curly braces doubled.
END_OF_PROMPT_INSTRUCTIONS = """
- Return only a valid JSON object with two keys:
"answer": must be "TRUE" or "FALSE" (all caps)
"relevant_cells": a list of objects, each with "row_index" (int) and "column_name" (string)

For example:

{{
  "answer": "TRUE",
  "relevant_cells": [
    {{"row_index": 0, "column_name": "Revenue"}},
    {{"row_index": 1, "column_name": "Employees"}}
  ]
}}

No extra keys, no extra text. Just that JSON. You are not supposed to provide any python code.
"""

ZERO_SHOT_PROMPT = """
You are tasked with determining whether a claim about the following table (given in {format_type} format) is TRUE or FALSE.

#### Table ({format_type}):
{table_formatted}

#### Claim:
"{claim}"

Instructions:
- Carefully check each condition in the claim against the table and determine which cells are relevant to verify the claim.
- If fully supported, answer "TRUE"; otherwise, answer "FALSE".
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


# chain-of-thought reasoning
COT_PROMPT_TEMPLATE = """
You are tasked with determining whether a claim about the following table (in {format_type} format) is TRUE or FALSE.
Before providing your final answer, explain step-by-step your reasoning process by referring to the relevant parts of the table.

#### Table ({format_type}):
{table_formatted}

#### Claim:
"{claim}"

Instructions:
- First, list your reasoning steps in a clear and logical order.
- After your explanation, output a final answer in a valid JSON object with the following format:
{{
  "chain_of_thought": "<your step-by-step reasoning here>",
  "answer": "TRUE" or "FALSE",
  "relevant_cells": [ list of relevant cells as objects with "row_index" and "column_name" ]
}}

Make sure that your output is strictly in this JSON format and nothing else.
"""

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
            prompt = ZERO_SHOT_PROMPT.format(
                table_formatted=table_formatted,
                claim=claim,
                format_type=self.format_type
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
        elif self.learning_type == "chain_of_thought":
            prompt = COT_PROMPT_TEMPLATE.format(
                table_formatted=table_formatted,
                claim=claim,
                format_type=self.format_type
            )
        else:
            logging.error(f"Unsupported learning type: {self.learning_type}")
            return None
        return prompt.strip()

################################################################################
#                         PIPELINE FUNCTION
################################################################################

def run_pipeline_for_model(model_name: str, dataset: str, learning_type: str, format_type: str, args):
    """
    Run the entire prompt engineering pipeline for one model and one combination
    of dataset, learning type, and format type.
    """
    repo_folder = args.repo_folder
    csv_folder = os.path.join(repo_folder, args.csv_folder)
    dataset_file = os.path.join(repo_folder, dataset)
    dataset_data = load_json_file(dataset_file)

    # Define a checkpoint folder specific to this configuration.
    config_str = f"{os.path.basename(dataset).replace('.json','')}_{learning_type}_{format_type}_{model_name}"
    checkpoint_folder = os.path.join("checkpoints_promptEngineering", config_str)
    
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
        learning_type=learning_type,
        format_type=format_type,
        model=model,
        model_name=model_name
    )
    
    # Process claims using batch mode if requested.
    if args.batch_prompts:
        results = test_model_on_claims_parallel(
            fact_checker_class=PromptEngineeringFactChecker,
            model_name=model_name,
            full_cleaned_data=dataset_data,
            all_csv_folder=csv_folder,
            learning_type=learning_type,
            format_type=format_type,
            approach="prompt_engineering",
            test_all=False if args.N else True,
            N=args.N,
            max_workers=args.max_workers,
            checkpoint_folder=checkpoint_folder
        )
    else:
        results = test_model_on_claims(
            fact_checker=fact_checker,
            full_cleaned_data=dataset_data,
            test_all=False if args.N else True,
            N=args.N,
            checkpoint_folder=checkpoint_folder
        )
    
    # Create output folder names that incorporate the combination.
    combo_str = f"{os.path.basename(dataset).replace('.json','')}_{learning_type}_{format_type}_{model_name}"
    results_folder = f"results_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(results_folder, f"results_prompt_engineering_{combo_str}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_file}")

    # also save final results to docs/
    website_results_folder = "../docs/results"
    os.makedirs(website_results_folder, exist_ok=True)
    with open(f"{website_results_folder}/results_with_cells_{model_name}_{os.path.basename(dataset).replace('.json','')}_{args.N if args.N else 'all'}_{learning_type}_{format_type}.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Wrote results to {website_results_folder}/results_with_cells_{model_name}_{os.path.basename(dataset).replace('.json','')}_{args.N if args.N else 'all'}_{learning_type}_{format_type}.json")

    
    metrics_folder = os.path.join(results_folder, f"plots_prompt_engineering_{combo_str}")
    os.makedirs(metrics_folder, exist_ok=True)
    calculate_and_plot_metrics(
        results=results,
        save_dir=metrics_folder,
        learning_type=learning_type,
        dataset_type=os.path.basename(dataset).replace('.json',''),
        model_name=model_name,
        format_type=format_type
    )

################################################################################
#                              MAIN FUNCTION
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Run Prompt Engineering Fact Checking Approach")
    parser.add_argument("--repo_folder", type=str, default="../original_repo", help="Path to repository folder")
    parser.add_argument("--csv_folder", type=str, default="data/all_csv/", help="Folder containing CSV tables")
    # Accept comma-separated lists for dataset, learning type, and format type.
    parser.add_argument("--dataset", type=str, default="tokenized_data/test_examples.json", help="Comma-separated list of dataset JSON files")
    parser.add_argument("--learning_type", type=str, default="zero_shot", help="Comma-separated list of learning types (e.g. zero_shot,one_shot,few_shot)")
    parser.add_argument("--format_type", type=str, default="naturalized", help="Comma-separated list of format types (e.g. markdown,naturalized, json, html)")
    parser.add_argument("--models", type=str, default="mistral", help="Comma-separated list of model names")
    parser.add_argument("--parallel_models", action="store_true", help="Run different combinations in parallel")
    parser.add_argument("--batch_prompts", action="store_true", help="Process claims in parallel for each combination")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of worker processes for parallel claims")
    parser.add_argument("--N", type=int, help="Number of tables to test") # if not provided, defaults to doing all
    args = parser.parse_args()
    
    # running on host/machine
    logging.info(f"Running on host/machine: {os.uname().nodename}")

    # Capture and log the command used to run the script
    command_used = " ".join(sys.argv)
    logging.info(f"Command used to run the script: {command_used}")


    # Split comma-separated arguments.
    dataset_list = [d.strip() for d in args.dataset.split(",")]
    learning_types = [l.strip() for l in args.learning_type.split(",")]
    format_types = [f.strip() for f in args.format_type.split(",")]
    model_list = [m.strip() for m in args.models.split(",")]
    
    # Build a list of all combinations.
    tasks = []
    for model_name in model_list:
        for dataset in dataset_list:
            for lt in learning_types:
                for ft in format_types:
                    tasks.append((model_name, dataset, lt, ft))
    
    # Run tasks in parallel or sequentially.
    if args.parallel_models:
        with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
            futures = []
            for (model_name, dataset, lt, ft) in tasks:
                futures.append(executor.submit(run_pipeline_for_model, model_name, dataset, lt, ft, args))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error in parallel execution: {e}")
    else:
        for (model_name, dataset, lt, ft) in tasks:
            run_pipeline_for_model(model_name, dataset, lt, ft, args)

    
    # write manifest file to docs/
    manifest_file = "../docs/results/manifest.json"
    os.makedirs("../docs/results", exist_ok=True)
    result_files = [
        f"results/{f}" for f in os.listdir("../docs/results")
        if f.startswith("results_with_cells_") and f.endswith(".json")
    ]
    manifest = {
        "results_files": result_files
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest generated with {len(result_files)} files.")

if __name__ == "__main__":
    main()
