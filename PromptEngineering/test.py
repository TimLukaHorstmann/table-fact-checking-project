import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

docs_results = "docs/results/results_with_cells_llama3.2:latest_test_examples_1695_zero_shot_html.json"

def load_predictions(file_path):
    """
    Load a JSON file containing a list of dictionaries into a Python list of dictionaries.
    
    """
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        print(f"Error loading JSON file from {file_path}: {e}")
        return []
    
claims = load_predictions(docs_results)
for claim in claims:
    print(claim['claim'])
    print()