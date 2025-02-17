#!/usr/bin/env python3
"""
code_generation_approach.py

This script implements a stub for a code generation based fact checker.
The generate_prompt() method should eventually instruct the LLM to produce executable code
(e.g. in pandas) that extracts or computes evidence from the table.
This script also supports both claim-level and model-level parallelization over multiple datasets,
learning types, and format types.
"""

import os
import json
import logging
import argparse
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, Any, List
import time
import sqlite3
from io import StringIO
import pandas as pd


from factchecker_base2 import BaseFactChecker, test_model_on_claims, test_model_on_claims_parallel, calculate_and_plot_metrics, load_json_file, format_table_to_json
from langchain_ollama import OllamaLLM


class CodeGenerationFactChecker(BaseFactChecker):
    """
    CodeGenerationFactChecker is a stub for a code generation approach.
    
    Extend this class to instruct the LLM to generate executable code (e.g., using pandas)
    that extracts or computes evidence from the table. This can involve:
      - Providing a code template in the prompt.
      - Specifying clear instructions on how the generated code should process the table.
      - Including an example of acceptable code output.
    """
    def __init__(self, *args, code_type="python", **kwargs):
        super().__init__(*args, **kwargs)
        self.code_type = code_type

    def generate_prompt(self, table_id: str, claim: str) -> Optional[str]:
        # logging.info("Code generation approach stub: Implement code-generation instructions here.")
        # Provide instructions to the LLM to output executable code.
        # For instance, "Write a Python function using pandas to filter rows where the claim holds..."
        df_table = self.load_table(table_id)
        json_table = format_table_to_json(df_table)
        column_types = df_table.dtypes.astype(str).to_dict()
        # /*
        #     "{df_table}.head(5).to_string(index=False)" as follows:
        #     {df_table.head(5).to_string(index=False)}
        # */
        if self.code_type == "python":
            prompt = f"""
            You are an AI assistant that writes Python code to check factual claims against tabular data.
            Given a json_table and a claim, generate a Python function 'check_claim(df)' using pandas that:
            - Assumes table is a pandas dataframe called df, corresponding to the json_table.
            - Filters rows of df that are relevant to the claim (look carefully at the column names).
            - Computes necessary statistics or extracts relevant values.
            - Returns a boolean indicating if the claim is supported by the table data.

            Here is the json_table:
            /*
                {json_table}
            */
            Claim on table: {claim}

            For additional context, here are the data types of each column in the table:
            /*
                {column_types}
            */

            

            Format of what you should ouput:
            ```python
            def check_claim(df):
                # Implement filtering logic
                # Implement claim validation logic
                return True or False  # Based on 
            ```
            Ensure strict type consistency when processing data. Pay close attention to whether values are strings, integers, booleans, or other types to prevent type-related errors.
            Output only Python code, encapsulated in ```python [code] ```, no explanations, no example usage..
            """
        elif self.code_type == "sql":
            prompt = f"""
                You are an AI assistant that writes SQL queries to check factual claims against tabular data.
                Generate a SQL query that:
                - Operates on a table named 'data' with columns: {list(df_table.columns)}
                - Returns 1 if claim is supported, 0 otherwise
                - Uses SQLite syntax

                Here is the table in json format:
                /*
                    {json_table}
                */
                Data types: {column_types}
                Claim on table: {claim}

                Output only SQL code, encapsulated in ```sql [code] ```, no explanations.
                """
        # print(f"claim is -------------- {claim}")
        return prompt

    def parse_response(self, raw_response: str, table_id: str, claim: str) -> Dict[str, Any]:
        """
        ***Method OVERRIDEN from the BaseFactChecker class***
        Parse the LLM response to extract the JSON output.
        """
        df = self.load_table(table_id)
        if self.code_type == "python":
            extracted_code = re.findall(r"```python(.*?)```", raw_response, re.DOTALL)
            if not extracted_code:
                logging.error("No code found in llm answer, for")
                is_supported = False  # No code block found
            else:
                code = "import pandas as pd\n" + extracted_code[-1].strip()
            try:
                exec_locals = {"df": df}
                exec(code, {}, exec_locals)
                check_function = exec_locals.get("check_claim")
                if check_function and callable(check_function):
                    is_supported = check_function(df)
                else:
                    logging.error("Generated code does not contain a valid check_claim function.")
                    is_supported = False
                answer = "TRUE" if is_supported else "FALSE"
                relevant_cells = []

            except Exception as e:
                logging.warning(f"Error processing claim with generated code: {e}, \n table and claim was {self.load_table(table_id) + claim} \nllm code was {code} ")
                answer = "FALSE"
                relevant_cells = []
            return {"answer": answer, "relevant_cells": relevant_cells}
        
        elif self.code_type == "sql":
            extracted_sql = re.findall(r"```sql\n?(.*?)\n?```", raw_response, re.DOTALL)
            
            # Fallback: look for bare SQL without code blocks
            if not extracted_sql:
                extracted_sql = re.findall(r"(SELECT .*?;)", raw_response, re.DOTALL | re.IGNORECASE)

            if not extracted_sql:
                logging.error("No SQL found in response")
                return {"answer": "FALSE", "relevant_cells": []}

            # Clean up the SQL query
            sql_query = extracted_sql[-1].strip()
            sql_query = re.sub(r'[\n\r]+', ' ', sql_query)
            sql_query = re.sub(r'\s+', ' ', sql_query)
            
            conn = sqlite3.connect(":memory:")
            df.to_sql('data', conn, index=False)
            
            # Validate SQL against table schema
            cursor = conn.cursor()
            try:
                cursor.execute("EXPLAIN " + sql_query)
            except sqlite3.Error as e:
                logging.error(f"Invalid SQL query: {e}\nQuery: {sql_query}")
                return {"answer": "FALSE", "relevant_cells": []}
            
            # Execute valid query
            result = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            # Handle results
            if not result.empty:
                # Look for any truthy value in first column
                is_supported = any(bool(x) for x in result.iloc[:, 0])
            else:
                is_supported = False
            
            return {"answer": "TRUE" if is_supported else "FALSE", "relevant_cells": []}

    def process_claim(self, table_id: str, claim: str, true_label: int) -> Dict[str, Any]:
        """
        ***Method OVERRIDEN from the BaseFactChecker class***
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
        parsed = self.parse_response(raw_response, table_id, claim)
        predicted_label = 1 if parsed.get("answer") == "TRUE" else 0

        ###### temp ####
        print(f"true={true_label}, pred={predicted_label}")
        ################
        return {
            "table_id": table_id,
            "claim": claim,
            "predicted_response": predicted_label,
            "resp": raw_response,
            "true_response": true_label,
            "relevant_cells": parsed.get("relevant_cells")
        }        


    

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

    fact_checker = CodeGenerationFactChecker(
        all_csv_folder=csv_folder,
        learning_type=learning_type,
        format_type=format_type,
        code_type=args.code_type,
        model=model
    )
    
    if args.batch_prompts:
        results = test_model_on_claims_parallel(
            fact_checker_class=CodeGenerationFactChecker,
            model_name=model_name,
            full_cleaned_data=dataset_data,
            all_csv_folder=csv_folder,
            learning_type=learning_type,
            format_type=format_type,
            approach="code_generation",
            test_all=args.test_all,
            N=args.N,
            max_workers=args.max_workers,
            code_type=getattr(args, 'code_type', 'python')

        )
    else:
        results = test_model_on_claims(
            fact_checker=fact_checker,
            full_cleaned_data=dataset_data,
            test_all=args.test_all,
            code_type=args.code_type,
            N=args.N,
            checkpoint_file=None
        )
    
    combo_str = f"{os.path.basename(dataset).replace('.json','')}_{fact_checker.code_type}_{learning_type}_{format_type}_{model_name}"
    results_folder = f"results_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(results_folder, f"results_CodeGeneration_{combo_str}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_file}")
    print(f"Results saved to {results_file}")
    
    metrics_folder = os.path.join(results_folder, f"plots_CodeGeneration_{combo_str}")
    os.makedirs(metrics_folder, exist_ok=True)
    calculate_and_plot_metrics(
        results=results,
        save_dir=metrics_folder,
        learning_type=learning_type,
        dataset_type=os.path.basename(dataset).replace('.json',''),
        model_name=model_name,
        format_type=format_type
    )

    #################################
    with open(results_file, "r") as f:
        results = json.load(f)

    correct_predictions = 0
    total_predictions = 0
    
    # Count correct predictions and total predictions
    for result in results:
        true_label = result['true_response']
        predicted_label = result['predicted_response']
        
        if predicted_label == true_label:
            correct_predictions += 1
        total_predictions += 1
    
    # Print out the results
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {correct_predictions / total_predictions * 100:.2f}%")


def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Run Code Generation Fact Checking Approach (stub)")
    parser.add_argument("--repo_folder", type=str, default="../original_repo", help="Path to repository folder")
    parser.add_argument("--csv_folder", type=str, default="data/all_csv/", help="Folder containing CSV tables")
    parser.add_argument("--dataset", type=str, default="tokenized_data/test_examples.json", help="Comma-separated list of dataset JSON files")
    parser.add_argument("--learning_type", type=str, default="zero_shot", help="Comma-separated list of learning types")
    parser.add_argument("--format_type", type=str, default="naturalized", help="Comma-separated list of format types")
    parser.add_argument("--models", type=str, default="mistral", help="Comma-separated list of model names")
    parser.add_argument("--parallel_models", action="store_true", help="Run different combinations in parallel")
    parser.add_argument("--batch_prompts", action="store_true", help="Process claims in parallel for each combination")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of worker processes for parallel claims")
    parser.add_argument("--test_all", type=bool, default=False, help="Wether to test all of the claims of the file or not")
    parser.add_argument("--N", type=int, default=5, help="Number of tables to test")
    parser.add_argument("--code_type", type=str, choices=["python", "sql"], default="python", help="Type of code to generate (python|sql)")
    
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
    end_time = time.time()
    print("total exec time=", round(end_time-start_time, 1), "seconds")
    



if __name__ == "__main__":
    main()
