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
import numpy as np

from factchecker_base import BaseFactChecker, test_model_on_claims, test_model_on_claims_parallel, calculate_and_plot_metrics, load_json_file
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings

class RAGFactChecker(BaseFactChecker):
    """
    RAGFactChecker implements a Retrieval-Augmented Generation approach.
    
    To extend this stub:
      - Implement table encoding to transform the table into a searchable or vectorized format.
      - Retrieve relevant passages or cells.
      - Integrate the retrieved evidence into the prompt before querying the LLM.
    """
    def generate_prompt(self, table_id: str, claim: str) -> Optional[str]:
        logging.info("RAG approach stub: Implement retrieval and reasoning here.")
        # TODO: 1) Load and encode the table data.
        #       2) Retrieve related evidence based on the claim.
        #       3) Format a prompt that combines the table, retrieved evidence, and the claim.
        logging.info("RAG approach (Row and Column): Generating prompt using both row and column embeddings with cosine similarity thresholds.")

        # Load the table.
        table = self.load_table(table_id)
        if table is None:
            logging.error(f"Failed to load table {table_id}.")
            return None

        # Prepare row texts.
        row_texts = []
        for _, row in table.iterrows():
            row_text = ", ".join(f"{col}: {row[col]}" for col in table.columns)
            row_texts.append(row_text)
            
        # Prepare column texts by combining the column name with all its values,
        # each on a new line.
        column_texts = []
        for col in table.columns:
            values = table[col].tolist()
            # Create a string with the column name followed by each value on a new line.
            col_text = f"{col}\n" + "\n".join(str(v) for v in values)
            column_texts.append(col_text)

        # Create an embeddings object using OllamaEmbeddings.
        from langchain_community.embeddings import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Generate embeddings for rows and columns.
        row_embeddings = embeddings.embed_documents(row_texts)
        column_embeddings = embeddings.embed_documents(column_texts)
        
        # Generate an embedding for the claim.
        claim_embedding = embeddings.embed_query(claim)

        # Compute cosine similarities.
        def cosine_similarity(a, b):
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

        row_similarities = [cosine_similarity(claim_embedding, emb) for emb in row_embeddings]
        threshold_row = 0.5  # adjust as needed
        selected_row_indices = [i for i, sim in enumerate(row_similarities) if sim >= threshold_row]
        if not selected_row_indices:
            selected_row_indices = [int(np.argmax(row_similarities))]
        relevant_rows = [row_texts[i] for i in selected_row_indices]

        column_similarities = [cosine_similarity(claim_embedding, emb) for emb in column_embeddings]
        threshold_col = 0.5  # adjust as needed
        selected_column_indices = [i for i, sim in enumerate(column_similarities) if sim >= threshold_col]
        if not selected_column_indices:
            selected_column_indices = [int(np.argmax(column_similarities))]
        
        relevant_columns = [column_texts[i] for i in selected_column_indices]

        # Build the prompt.
        prompt = f"Claim: {claim}\n\nRelevant Table Rows:\n"
        for idx, row in enumerate(relevant_rows, start=1):
            prompt += f"-> {row}\n"
        prompt += "\nRelevant Table Columns:\n"
        for idx, col in enumerate(relevant_columns, start=1):
            prompt += f"----------\ncolumn name: {col}\n"
        prompt += "\nBased on the above information, please determine if the claim is true or false, and provide a brief explanation."
        prompt += "\nAfter the explanation provide the answer in the following JSON format:\n\n" \
                "```json\n" \
                "{\n" \
                '    "answer": "TRUE"  # or "FALSE"\n' \
                "}\n" \
                "```\n"
        print(prompt)

        return prompt



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
            checkpoint_folder=None
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
