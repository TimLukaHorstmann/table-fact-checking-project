#!/usr/bin/env python3
"""
rag_approach.py

This script implements a stub for a Retrieval-Augmented Generation (RAG) based fact checker.
Extend the generate_prompt() method with your RAG-specific logic (e.g. table encoding, retrieval,
and reasoning). This script also supports parallel processing over multiple datasets,
learning types, and format types.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from factchecker_base import BaseFactChecker, test_model_on_claims, test_model_on_claims_parallel, calculate_and_plot_metrics, load_json_file
from langchain_ollama import OllamaLLM

class RAGFactChecker(BaseFactChecker):
    def generate_prompt(self, table_id: str, claim: str) -> Optional[str]:
        logging.info("RAG approach stub: Implement retrieval and reasoning here.")
        # TODO: Implement table encoding, retrieval, and reasoning logic.
        return None

def run_pipeline_for_model(model_name: str, dataset: str, learning_type: str, format_type: str, args):
    repo_folder = args.repo_folder
    csv_folder = os.path.join(repo_folder, args.csv_folder)
    dataset_file = os.path.join(repo_folder, dataset)
    dataset_data = load_json_file(dataset_file)
    
    try:
        if "deepseek" in model_name:
            model = OllamaLLM(model=model_name, params={"n_ctx": 4096, "n_batch": 256})
        else:
            model = OllamaLLM(model=model_name)
    except Exception as e:
        logging.error(f"Failed to initialize model {model_name}: {e}")
        return

    fact_checker = RAGFactChecker(
        all_csv_folder=csv_folder,
        learning_type=learning_type,
        format_type=format_type,
        model=model
    )
    
    if args.batch_prompts:
        results = test_model_on_claims_parallel(
            fact_checker_class=RAGFactChecker,
            model_name=model_name,
            full_cleaned_data=dataset_data,
            all_csv_folder=csv_folder,
            learning_type=learning_type,
            format_type=format_type,
            approach="rag",
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
    
    combo_str = f"{os.path.basename(dataset).replace('.json','')}_{learning_type}_{format_type}_{model_name}"
    results_folder = f"results_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(results_folder, f"results_RAG_{combo_str}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_file}")
    
    metrics_folder = os.path.join(results_folder, f"plots_RAG_{combo_str}")
    os.makedirs(metrics_folder, exist_ok=True)
    calculate_and_plot_metrics(
        results=results,
        save_dir=metrics_folder,
        learning_type=learning_type,
        dataset_type=os.path.basename(dataset).replace('.json',''),
        model_name=model_name,
        format_type=format_type
    )

def main():
    parser = argparse.ArgumentParser(description="Run RAG Fact Checking Approach (stub)")
    parser.add_argument("--repo_folder", type=str, default="../original_repo", help="Path to repository folder")
    parser.add_argument("--csv_folder", type=str, default="data/all_csv/", help="Folder containing CSV tables")
    parser.add_argument("--dataset", type=str, default="tokenized_data/test_examples.json", help="Comma-separated list of dataset JSON files")
    parser.add_argument("--learning_type", type=str, default="zero_shot", help="Comma-separated list of learning types")
    parser.add_argument("--format_type", type=str, default="naturalized", help="Comma-separated list of format types")
    parser.add_argument("--models", type=str, default="mistral", help="Comma-separated list of model names")
    parser.add_argument("--parallel_models", action="store_true", help="Run different combinations in parallel")
    parser.add_argument("--batch_prompts", action="store_true", help="Process claims in parallel for each combination")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of worker processes for parallel claims")
    parser.add_argument("--N", type=int, default=5, help="Number of tables to test")
    args = parser.parse_args()
    
    dataset_list = [d.strip() for d in args.dataset.split(",")]
    learning_types = [l.strip() for l in args.learning_type.split(",")]
    format_types = [f.strip() for f in args.format_type.split(",")]
    model_list = [m.strip() for m in args.models.split(",")]
    
    tasks = []
    for model_name in model_list:
        for dataset in dataset_list:
            for lt in learning_types:
                for ft in format_types:
                    tasks.append((model_name, dataset, lt, ft))
    
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

if __name__ == "__main__":
    main()
