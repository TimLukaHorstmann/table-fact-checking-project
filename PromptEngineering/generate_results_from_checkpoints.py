#!/usr/bin/env python3
"""
generate_results_from_checkpoints.py

This script scans the checkpoints folder (which is assumed to have subfolders for each
configuration, named as:
    {dataset_basename}_{learning_type}_{format_type}_{model_name}
It combines all the JSON checkpoint files (each representing one table) from each configuration
into a single result file and writes that file to the docs/results folder using the filename format:
    results_with_cells_{model_name}_{dataset_basename}_{N}_{learning_type}_{format_type}.json
where N is the number of tables (i.e. checkpoint files) in that configuration.

It then creates a manifest file listing all result files.
"""

import os
import json
import logging
from datetime import datetime

# Configure logging for this script.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_results_from_checkpoints(
    checkpoints_root: str = "checkpoints_promptEngineering",
    docs_results_folder: str = "../docs/results"
):
    """
    For each configuration folder in checkpoints_root, read all checkpoint JSON files,
    combine them, and write a results file to docs_results_folder.
    Then, generate a manifest file listing all result files.
    """
    if not os.path.exists(checkpoints_root):
        logging.error(f"Checkpoints folder '{checkpoints_root}' does not exist.")
        return

    os.makedirs(docs_results_folder, exist_ok=True)
    result_files = []  # List of result file names (with the "results/" prefix)

    # Iterate over each configuration folder in checkpoints_root.
    for config_name in os.listdir(checkpoints_root):
        config_path = os.path.join(checkpoints_root, config_name)
        if not os.path.isdir(config_path):
            continue

        combined_results = []
        table_files = [f for f in os.listdir(config_path) if f.endswith(".json")]
        N = len(table_files)
        if N == 0:
            logging.warning(f"No checkpoint JSON files found in {config_path}. Skipping.")
            continue

        # Read each checkpoint file and add its results.
        for fname in table_files:
            file_path = os.path.join(config_path, fname)
            try:
                with open(file_path, "r") as f:
                    table_results = json.load(f)
                combined_results.extend(table_results)
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")
                continue

        # Our config_name is expected to be in the format:
        #    {dataset_basename}_{learning_type}_{format_type}_{model_name}
        # we must note, however, that the individual elements can contain underscores!
        parts = config_name.split("_")

        dataset_basename = parts[0] + "_" + parts[1]  # We combine the first two parts
        model_name = parts[-1]
        format_type = parts[-2]

        # rest of the parts are learning_type and depending on the number of parts, we can determine the learning_type
        learning_type = "_".join(parts[2:-2])

        print(f"dataset_basename: {dataset_basename}")
        print(f"model_name: {model_name}")
        print(f"learning_type: {learning_type}")
        print(f"format_type: {format_type}")


        # Ensure that the dataset_basename is one of the expected ones (e.g., test_examples or val_examples).
        # If not, you might want to adjust or skip.
        if dataset_basename not in ["test_examples", "val_examples"]:
            logging.warning(f"Dataset basename '{dataset_basename}' not recognized (expected 'test_examples' or 'val_examples'). Skipping {config_name}.")
            continue

        # Create the result file name as expected by the website:
        # results_with_cells_{model_name}_{dataset_basename}_{N}_{learning_type}_{format_type}.json
        result_filename = f"results_with_cells_{model_name}_{dataset_basename}_{N}_{learning_type}_{format_type}.json"
        result_filepath = os.path.join(docs_results_folder, result_filename)
        try:
            with open(result_filepath, "w") as f:
                json.dump(combined_results, f, indent=2)
            logging.info(f"Written combined results for config '{config_name}' ({N} tables) to {result_filepath}")
            # The manifest expects file paths starting with "results/"
            result_files.append(f"results/{result_filename}")
        except Exception as e:
            logging.error(f"Error writing {result_filepath}: {e}")

    # Write out the manifest file.
    manifest = {"results_files": result_files}
    manifest_filepath = os.path.join(docs_results_folder, "manifest.json")
    try:
        with open(manifest_filepath, "w") as f:
            json.dump(manifest, f, indent=2)
        logging.info(f"Manifest file written to {manifest_filepath} with {len(result_files)} result files.")
    except Exception as e:
        logging.error(f"Error writing manifest file {manifest_filepath}: {e}")

if __name__ == "__main__":
    generate_results_from_checkpoints()
