import json
import os

def load_claims(file_path: str) -> list:
    """
    Load a JSON file from the given path and return a list of dictionaries with 'claim' and 'ID' 
    for each element.
    """
    all_claims = []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        id_counter = 1
        
        for item in data:
            claim_with_resp = {
                "ID": id_counter,
                "claim": item.get("claim", ""),
            }
            
            id_counter += 1
            
            all_claims.append(claim_with_resp)
    
    except Exception as e:
        print(f"Error loading file: {e}")
    
    return all_claims

import os

def list_files_in_folder(path_name: str) -> list:
    """
    Given a folder path, return a list of all file names in the folder.

    Parameters:
    - path_name: The path to the folder.

    Returns:
    - A list of file names in the folder.
    """
    try:
        # List all files in the given folder
        file_names = [f for f in os.listdir(path_name) if os.path.isfile(os.path.join(path_name, f))]
        return file_names
    except Exception as e:
        print(f"Error listing files in folder: {e}")
        return []


path_name = "docs/results/"
l = list_files_in_folder(path_name)

print(l)
for file in l:
    d = load_claims("docs/results/" + file)
    print(file, len(d))

print(len(load_claims("PromptEngineering/results_20250212/results_CodeGeneration_test_examples_zero_shot_naturalized_phi4.json")))