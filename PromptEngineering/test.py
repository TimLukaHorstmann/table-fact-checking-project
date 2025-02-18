import json

# Load the JSON data from the files

with open('PromptEngineering/Error_Analysis/our_category_claims_mistral.json', 'r') as file_mistral:
    mistral_data = json.load(file_mistral)

with open('PromptEngineering/Error_Analysis/our_category_claims_bis_mistral.json', 'r') as file_bis_mistral:
    bis_mistral_data = json.load(file_bis_mistral)

# Calculate the lengths of both lists
len_mistral = len(mistral_data)
len_bis_mistral = len(bis_mistral_data)

# Print the lengths
print(f"Length of our_category_claims_mistral.json: {len_mistral}")
print(f"Length of our_category_claims_bis_mistral.json: {len_bis_mistral}")

# Extract the 'claim' values from both lists
mistral_claims = {item['claim'] for item in mistral_data}
bis_mistral_claims = {item['claim'] for item in bis_mistral_data}

# Find the missing claims from 'bis_mistral' that are in 'mistral'
missing_claims = mistral_claims - bis_mistral_claims

# Print the missing claims
print(f"Missing claims in our_category_claims_bis_mistral.json:")
for claim in missing_claims:
    print(claim)