import os
import json
import logging
from datetime import datetime
from tqdm import tqdm

# Configure logging for this script.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Path to the simple and complex tables IDs
SIMPLE_IDS_FILE = "../original_repo/data/simple_ids.json"
COMPLEX_IDS_FILE = "../original_repo/data/complex_ids.json"

def load_table_ids():
    """
    Load the table IDs from simple_ids.json and complex_ids.json.
    Returns a dictionary of table IDs classified as simple or complex.
    """
    simple_ids = set()
    complex_ids = set()
    
    try:
        with open(SIMPLE_IDS_FILE, "r") as f:
            simple_ids = set(json.load(f))
    except Exception as e:
        logging.error(f"Error reading {SIMPLE_IDS_FILE}: {e}")
    
    try:
        with open(COMPLEX_IDS_FILE, "r") as f:
            complex_ids = set(json.load(f))
    except Exception as e:
        logging.error(f"Error reading {COMPLEX_IDS_FILE}: {e}")
    
    return simple_ids, complex_ids

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

    # Load simple and complex table IDs
    simple_ids, complex_ids = load_table_ids()

    # Iterate over each configuration folder in checkpoints_root.
    for config_name in tqdm(os.listdir(checkpoints_root)):
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
                
                # Check if the table is simple or complex and add this info to each table's result
                for result in table_results:
                    table_id = result.get("table_id")  # Assuming the table ID is stored in the "table_id" key
                    if table_id in simple_ids:
                        result["table_type"] = "simple"
                    elif table_id in complex_ids:
                        result["table_type"] = "complex"
                    else:
                        result["table_type"] = "unknown"  # Default case, if not found

                combined_results.extend(table_results)
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")
                continue

        # Our config_name is expected to be in the format:
        #    {dataset_basename}_{learning_type}_{format_type}_{model_name}
        parts = config_name.split("_")

        dataset_basename = parts[0] + "_" + parts[1]  # We combine the first two parts
        model_name = parts[-1]
        format_type = parts[-2]

        learning_type = "_".join(parts[2:-2])

        # Ensure that the dataset_basename is one of the expected ones (e.g., test_examples or val_examples).
        if dataset_basename not in ["test_examples", "val_examples"]:
            logging.warning(f"Dataset basename '{dataset_basename}' not recognized (expected 'test_examples' or 'val_examples'). Skipping {config_name}.")
            continue

        # Create the result file name as expected by the website:
        result_filename = f"results_with_cells_{model_name}_{dataset_basename}_{N}_{learning_type}_{format_type}.json"
        result_filepath = os.path.join(docs_results_folder, result_filename)
        try:
            with open(result_filepath, "w") as f:
                json.dump(combined_results, f, indent=2)
            logging.info(f"Written combined results for config '{config_name}' ({N} tables) to {result_filepath}")
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