import json
import os

def load_claims(file_path: str) -> list:
    """
    Load a JSON file from the given path and return a list of dictionaries with 'claim' and 'ID' 
    for each element.
    """
    all_claims = []
    
    try:
        with open("docs/results/" + file_path, 'r') as f:
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





a = "results_with_cells_deepseek-r1:8b_test_examples_1694_zero_shot_html.json"
b = "results_with_cells_llama3.2:latest_test_examples_46_chain_of_thought_json.json"
c = "results_with_cells_mistral:latest_test_examples_1695_zero_shot_naturalized.json"

d = load_claims(a)
e = load_claims(b)
f = load_claims(c)

print(os.getcwd())
print(len(d), len(e), len(f))