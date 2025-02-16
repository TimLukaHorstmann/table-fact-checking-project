#!/miniconda3/envs/llm/bin/python
"""
Categorise_claims.py

This script categorizes all the claims from the data-set into the following categories:
1. Aggregation: the aggregation operation refers to sentences like “the averaged age of all....”, “the total amount of scores obtained in ...”, etc.
2. Negation: the negation operation refers to sentences like “xxx did not get the best score”, “xxx has never obtained a score higher than 5”.
3. Superlative: the superlative operation refers to sentences like “xxx achieves the highest score in”, “xxx is the lowest player in the team”.
4. Comparative: the comparative operation refers to sentences like “xxx has a higher score than yyy”.
5. Ordinal: the ordinal operation refers to sentences like “the first country to achieve xxx is xxx”, “xxx is the second oldest person in the country”.
6. Unique: the unique operation refers to sentences like “there are 5 different nations in the tournament, ”, “there are no two different players from U.S”
7. All: the for all operation refers to sentences like “all of the trains are departing in the morning”, “none of the people are older than 25.”
8. None: the sentences which do not involve higher-order operations like “xxx achieves 2 points in xxx game”, “xxx player is from xxx country”.
"""

from logging import config
import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from tqdm import tqdm
from langchain_ollama import OllamaLLM
import logging
import concurrent.futures

################################################################################
#                        LOAD CLAIMS
################################################################################

import json

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

################################################################################
#                        PROMPT GENERATORS FOR CLASSIFICATION
################################################################################

def template_generation(category_explanation: str, claim: str, category: str) -> str:
    """
    Generates a template for the LLM that explains the task and asks for the categorization decision.
    
    Arguments:
    - category_explanation: A detailed explanation of the category, providing context and guidance.
    - claim: The claim that needs to be categorized.
    - category: The specific category for which the claim is being assessed.
    
    Returns:
    - A string that serves as the prompt for the LLM.
    """
    
    # Start building the prompt
    prompt = f"""You are an expert in categorizing claims.

Your task:
- Determine if the following claim fits the category: "{category}".
- Base your decision only on the given category definition.
- Provide a concise reasoning before your final decision.

Category Definition:
{category_explanation}

Claim to Categorize:
"{claim}"

Your response format:
1. Short explanation for your decision.
2. Final answer on the last line, strictly as: TRUE or FALSE.

Final answer:
"""
    
    return prompt

###################      CATEGORIES FROM THE PAPER       ################################################################

def aggregation_prompt_generation(claim: str) -> str:
    """
    Generates a prompt for the LLM to categorize a claim as falling under the 'aggregation' category.
    
    Arguments:
    - claim: The claim to be assessed for aggregation.
    
    Returns:
    - A string that serves as the prompt for the LLM to evaluate the claim.
    """
    
    # Explanation for aggregation category
    category_explanation = """
    Aggregation refers to operations where data is combined or summarized into a total or an average.
    If the claim involves a sum, an average, or any operation aggregating verification, it falls under this category.
    For example, “the averaged age of all...”, “the total amount of scores obtained in ...” are in the category 'Aggregation'.
    """

    # Generate the prompt using the template_generation function
    return template_generation(category_explanation, claim, "aggregation")

def negate_prompt_generation(claim: str) -> str:
    """
    Generates a prompt for the LLM to categorize a claim as falling under the 'negation' category.
    
    Arguments:
    - claim: The claim to be assessed for negation.
    
    Returns:
    - A string that serves as the prompt for the LLM to evaluate the claim.
    """
    
    # Refined explanation for negation category
    category_explanation = """
    Negation refers to sentences that deny or contradict a claim or assertion. 
    These often include phrases such as "did not", "never", or "not", which indicate that the statement is asserting the absence or opposite of something.
    For example, “xxx did not get the best score”, “xxx has never obtained a score higher than 5” are in the category 'Negation'.
    """
    
    # Generate the prompt using the template_generation function
    return template_generation(category_explanation, claim, "negate")

def superlative_prompt_generation(claim: str) -> str:
    """
    Generates a prompt for the LLM to categorize a claim as falling under the 'superlative' category.
    
    Arguments:
    - claim: The claim to be assessed for superlative.
    
    Returns:
    - A string that serves as the prompt for the LLM to evaluate the claim.
    """
    
    # Refined explanation for superlative category
    category_explanation = """
    Superlative refers to sentences that express the highest degree or extreme of something. 
    This often involves words like "best", "most", "highest", "greatest", which indicate that the claim is about the extreme or top value in a given context.
    For example, “xxx achieves the highest score in”, “xxx is the lowest player in the team" are in the category 'Superlative'.
    """
    
    # Generate the prompt using the template_generation function
    return template_generation(category_explanation, claim, "superlative")

def comparative_prompt_generation(claim: str) -> str:
    """
    Generates a prompt for the LLM to categorize a claim as falling under the 'comparative' category.
    
    Arguments:
    - claim: The claim to be assessed for comparative.
    
    Returns:
    - A string that serves as the prompt for the LLM to evaluate the claim.
    """
    
    # Refined explanation for comparative category
    category_explanation = """
    Comparative refers to claims that make comparisons between two or more items or concepts. 
    These claims often include words like "better", "more", "greater", "worse", "less", "higher", etc., to indicate a comparison of relative values or qualities between different entities.
    For example, “xxx achieves the highest score in”, “xxx is the lowest player in the team" are in the category 'Comparative'.
    """
    
    # Generate the prompt using the template_generation function
    return template_generation(category_explanation, claim, "comparative")

def ordinal_prompt_generation(claim: str) -> str:
    """
    Generates a prompt for the LLM to categorize a claim as falling under the 'ordinal' category.
    
    Arguments:
    - claim: The claim to be assessed for ordinal.
    
    Returns:
    - A string that serves as the prompt for the LLM to evaluate the claim.
    """
    
    # Refined explanation for ordinal category
    category_explanation = """
    Ordinal refers to claims that involve rankings or positions in a sequence. These claims typically involve terms like "first", "second", "last", "top", "bottom", "ranked", and so on, to indicate a specific position or order in a list.
    For example, “the first country to achieve xxx is xxx”, “xxx is the second oldest person in the country” are in the category 'Ordinal'.
    """
    
    # Generate the prompt using the template_generation function
    return template_generation(category_explanation, claim, "ordinal")

def unique_prompt_generation(claim: str) -> str:
    """
    Generates a prompt for the LLM to categorize a claim as falling under the 'unique' category.
    
    Arguments:
    - claim: The claim to be assessed for uniqueness.
    
    Returns:
    - A string that serves as the prompt for the LLM to evaluate the claim.
    """
    
    # Refined explanation for unique category
    category_explanation = """
    Unique refers to claims that involve the distinctness or uniqueness of items or entities. These claims often use phrases like "different", "no two", "only one", "none alike", and "distinct". The claim may suggest that something is the only one of its kind or that no two items are the same in some context.
    For example, “there are 5 different nations in the tournament, ”, “there are no two different players from U.S” are in the category 'Unique'.
    """
    
    # Generate the prompt using the template_generation function
    return template_generation(category_explanation, claim, "unique")

def all_prompt_generation(claim: str) -> str:
    """
    Generates a prompt for the LLM to categorize a claim as falling under the 'all' category.
    
    Arguments:
    - claim: The claim to be assessed for being universally applicable.
    
    Returns:
    - A string that serves as the prompt for the LLM to evaluate the claim.
    """
    
    # Refined explanation for all category
    category_explanation = """
    The 'All' category refers to claims that involve statements about the entirety or totality of a set or group. These claims use phrases like "all of", "every", "none of", or "always". The claim might refer to the full set or group, with no exceptions or exclusions.
    For example, “all of the trains are departing in the morning”, “none of the people are older than 25” are in the category 'All'.
    """
    
    # Generate the prompt using the template_generation function
    return template_generation(category_explanation, claim, "all")

def none_prompt_generation(claim: str) -> str:
    """
    Generates a prompt for the LLM to categorize a claim as falling under the 'none' category.
    
    Arguments:
    - claim: The claim to be assessed for being a simple factual statement.
    
    Returns:
    - A string that serves as the prompt for the LLM to evaluate the claim.
    """
    
    # Refined explanation for none category
    category_explanation = """
    The 'None' category refers to claims that are simple statements without any complex operations such as sums, comparisons, or qualifications. These claims typically involve basic factual information about a subject, like a specific value or identity.
    For example, “xxx achieves 2 points in xxx game”, “xxx player is from xxx country” are in the category 'None'.
    """
    
    # Generate the prompt using the template_generation function
    return template_generation(category_explanation, claim, "none")

###################      OUR CATEGORIES      #############################################################################

def single_fact_verification_prompt_generation(claim: str) -> str:
    category_explanation = """
Single Attribute Verification checks if a specific fact about an entity (like a name, value, or label) matches the dataset. The model should determine whether the claim correctly states a single factual attribute.
Example: "Tony Martin be the name of the mountain classification." 
The model must verify if "Tony Martin" is indeed the correct classification name.
    """
    return template_generation(category_explanation, claim, "Single Attribute Verification")

def multi_fact_conjunction_prompt_generation(claim: str) -> str:
    category_explanation = """
Multi-Attribute Conjunction checks if multiple attributes of an entity are all correct. The model should assess whether all stated properties (e.g., name, price, year) match the dataset.
Example: "The amethyst’s composition be 99.99% silver with a price of 94.95 in 2008."
The model must verify if the composition, price, and year are accurate.
    """
    return template_generation(category_explanation, claim, "Multi-Attribute Conjunction")

def negation_absence_check_prompt_generation(claim: str) -> str:
    category_explanation = """
Negation/Absence Check determines if a fact or attribute is explicitly missing or does not exist in the dataset. The model should decide if the claim correctly asserts the absence of an entity, title, or event.
Example: "None of the team classification name be Team Saxo Bank."
The model must verify if "Team Saxo Bank" is absent from the dataset.
    """
    return template_generation(category_explanation, claim, "Negation/Absence Check")

def numerical_threshold_check_prompt_generation(claim: str) -> str:
    category_explanation = """
Numerical Threshold Check verifies if a numerical value in the claim satisfies a given condition (>, <, or =). The model should assess whether the stated number is correct in relation to the dataset.
Example: "The lowest ticket price be 17.50."
The model must check if 17.50 is indeed the lowest recorded ticket price.
    """
    return template_generation(category_explanation, claim, "Numerical Threshold Check")

def aggregation_computation_prompt_generation(claim: str) -> str:
    category_explanation = """
Aggregation & Computation checks if a claim about sums, averages, or totals is correct. The model must determine whether the stated value results from accurate calculations based on the dataset.
Example: "The average score for Japanese players was 281."
The model must verify if the computed average score matches the claim.
    """
    return template_generation(category_explanation, claim, "Aggregation & Computation")

def record_existence_prompt_generation(claim: str) -> str:
    category_explanation = """
Record Existence checks whether at least one matching record exists in the dataset. The model must determine if the claim correctly asserts the presence of an entity or attribute.
Example: "There be 1 centerfold in the 9-98 issue."
The model must verify if a matching entry exists in the dataset.
    """
    return template_generation(category_explanation, claim, "Record Existence")

def comparative_analysis_prompt_generation(claim: str) -> str:
    category_explanation = """
Comparative Analysis checks if one value is greater than, smaller than, or equal to another across different entities, locations, or time periods. The model must verify whether the comparison holds.
Example: "Ticket price in Oklahoma City be higher than in Omaha."
The model must compare the ticket prices in both locations and determine if the statement is correct.
    """
    return template_generation(category_explanation, claim, "Comparative Analysis")

def conditional_logic_prompt_generation(claim: str) -> str:
    category_explanation = """
Conditional Logic checks if a claim with dependencies holds true. The model should assess whether given conditions (X and Y) logically lead to a specific outcome (Z).
Example: "When play = 2 and 100+ = 36, the high checkout = 115."
The model must verify if the stated conditions correctly lead to the expected result.
    """
    return template_generation(category_explanation, claim, "Conditional Logic")



###################      OUR CATEGORIES BIS      #############################################################################


def unclear_or_noisy_language_prompt_generation(claim: str) -> str:
    category_explanation = """
Unclear or Noisy Language:  
This category includes claims that are difficult to interpret due to inconsistent language, noisy grammar, or ambiguous phrasing. Claims in this category may contain severe grammatical errors, missing words, awkward structure, or unclear references, making their meaning hard to determine.  

Examples:  
1. "jonathan legear score 4 more goal than matías suárez , the next highest rank player in the belgian first dvision a league who play in the belgian cup" – unclear structure makes it difficult to determine the goal count relationship.  
2. "the only tournament that tony lema win in be the open championship" – poor grammar makes it unclear if this is stating that he won only one tournament or something else.  
3. "natalia strelkova be not the female lose the skating championship" – severe grammatical issues make it ambiguous whether she did not lose or was not the only female who lost.  
    """
    return template_generation(category_explanation, claim, 'unclear_or_noisy_language')

def numerical_reasoning_prompt_generation(claim: str) -> str:
    category_explanation = """
Numerical Reasoning:
This category is for claims that involve numbers, calculations, or quantitative comparisons. The model should verify arithmetic relations, comparisons, or threshold conditions presented in the claim.

Examples:
1. "jonathan legear score 4 more goal than matías suárez" – requires comparing numerical values.
2. "pierre lamine have a mere 0.16 more point than shinji someya" – involves a precise numerical difference.
3. "the gap between first and last be a total of 58.04" – checking a numerical total.
    """
    return template_generation(category_explanation, claim, 'numerical_reasoning')

def multistep_logic_prompt_generation(claim: str) -> str:
    category_explanation = """
Multistep Logic:
This category is for claims that require processing several logical steps or conditions to be validated. The claim may involve a sequence of comparisons, conditions, or operations that need to be evaluated in order.

Examples:
1. "the term start for bashkim fino be after the term start for vilson ahmeti" – involves comparing two time points.
2. "when play = 2 and 100+ = 36, the high checkout = 115" – multiple conditions leading to an outcome.
3. "ettore meini win 3 race in a row, on may 24, 25th and 26th" – requires understanding sequential events.
    """
    return template_generation(category_explanation, claim, 'multistep_logic')


def negation_prompt_generation(claim: str) -> str:
    category_explanation = """
Negation:
This category applies to claims that contain explicit negation or denial. The model should flag claims that state something does not occur, using words like 'do not', 'not', or 'none', or otherwise imply absence.

Examples:
1. "tony lema do not win in the pga championship" – clearly denies a win.
2. "tournament that tony lema have not participate in include..." – indicates non-participation.
3. "neither team score for only the first game of the world cup in france" – explicitly denies scoring.
    """
    return template_generation(category_explanation, claim, 'negation')


###################      ONE CATEGORY PER CLAIM APPROACH      ############################################################


def one_category_per_claim_prompt_generation(claim):
    return f"""You are an expert in logical reasoning and claim classification.

Your task:
- Categorize the following claim into exactly one of the predefined categories.
- Justify your reasoning in 1-2 sentences.
- Provide your final answer on the last line in the format: 
  ANSWER: X (where X is the category number).

Categories:
1. Aggregation – The claim requires summing, averaging, or counting values.
   Example: "The average score of all players was 85."

2. Negation – The claim explicitly negates a fact.
   Example: "No player scored above 90."

3. Superlative – The claim involves the highest, lowest, best, or worst.
   Example: "The highest-scoring player was John."

4. Comparative – The claim compares two entities.
   Example: "Alice scored higher than Bob."

5. Ordinal – The claim ranks something in a sequence.
   Example: "This was the second-largest tournament in history."

6. Unique – The claim asserts that something is the only one of its kind.
   Example: "Only one team remained undefeated."

7. All – The claim makes a universal/general statement.
   Example: "All players participated in at least one game."

8. None – The claim is a factual statement that does not involve any higher-order logic.
   Example: "John scored 25 points in the match."

Claim to categorize:
{claim}

Your response format:
1. Short reasoning for your classification.
2. Final answer: ANSWER: X (where X is the category number).
"""

################################################################################
#                        CATEGORISE CLAIMS
################################################################################

def process_prediction(all_claims, i, prompt_generation_function, category, model):
    try:
        
        claim = all_claims[i]['claim']
        prompt = prompt_generation_function(claim)

        model_response = model.invoke(prompt)
        model_true = re.search(r"\bTRUE\b", model_response)


        if model_true:
            all_claims[i][category] = True

    except Exception as e:
        claim = all_claims[i]['claim']
        logging.error(f"Error processing claim {claim}: {e}")

def error_analysis_pipeline(all_claims, model_name, prompt_functions_with_categories):
    """
    Runs error analysis over failed and successful predictions, processing claims through various prompt generation functions.
    """
    try:
        model = OllamaLLM(model=model_name)
    except Exception as e:
        logging.error(f"Failed to initialize model {model_name}: {e}")
        return
    
    # Iterate through each prompt function and its associated category
    for prompt_function_name, category in prompt_functions_with_categories.items():
        print(f"Processing {prompt_function_name}")

        prompt_generation_function = globals().get(prompt_function_name)
        if not prompt_generation_function:
            # logging.error(f"Prompt generation function {prompt_function_name} not found.")
            continue  # Skip to the next prompt function if not found

        # Use ThreadPoolExecutor for parallel processing of claims
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            total_claims = len(all_claims)

            # Create a tqdm progress bar for claims processing
            with tqdm(total=total_claims, desc=f"Processing {category} claims", ncols=100, position=0) as pbar:
                # Submit tasks to the executor for processing each claim
                for i in range(total_claims):
                    futures.append(executor.submit(process_prediction, all_claims, i, prompt_generation_function, category, model))
                    
                # Wait for all futures to complete and update the progress bar
                for future in concurrent.futures.as_completed(futures):
                    # try:
                    future.result()  # Get result of each future, ensuring completion
                    # except Exception as e:
                        # logging.error(f"Error processing future: {e}")
                    pbar.update(1)  # Update the progress bar

    return all_claims





def one_category_process_prediction(all_claims, i, model):
    try:
        claim = all_claims[i]['claim']
        prompt = one_category_per_claim_prompt_generation(claim)

        model_response = model.invoke(prompt)
        last_line = model_response.strip().split("\n")[-1]  # Get last line
        answer_match = re.search(r"\d+", last_line)  # Extract the number at the end

        if answer_match:
            answer = int(answer_match.group())  # Convert to integer
        else:
            logging.warning(f"No valid category found for claim: {claim}")
            return

        # Map number to category
        category_map = {
            1: "aggregation",
            2: "negation",
            3: "superlative",
            4: "comparative",
            5: "ordinal",
            6: "unique",
            7: "all",
            8: "none"
        }

        if answer in category_map:
            all_claims[i][category_map[answer]] = True
        else:
            logging.warning(f"Invalid category number {answer} for claim: {claim}")

    except Exception as e:
        logging.error(f"Error processing claim {all_claims[i]['claim']}: {e}")

def one_category_error_analysis_pipeline(all_claims, model_name):
    """
    Runs error analysis over failed and successful predictions, processing each claim
    through the one_category_per_claim_prompt_generation function.
    """
    try:
        model = OllamaLLM(model=model_name)
    except Exception as e:
        logging.error(f"Failed to initialize model {model_name}: {e}")
        return
        
    # Use ThreadPoolExecutor for parallel processing of claims
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        total_claims = len(all_claims)

        # Create a tqdm progress bar for claims processing
        with tqdm(total=total_claims, desc="Processing claims", ncols=100, position=0) as pbar:
            # Submit tasks to the executor for processing each claim
            for i in range(total_claims):
                futures.append(executor.submit(one_category_process_prediction, all_claims, i, model))
                
            # Wait for all futures to complete and update the progress bar
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Ensure completion
                pbar.update(1)  # Update the progress bar

    return all_claims





################################################################################
#                        SAVE AND LOAD CLAIMS
################################################################################

def save_to_json(data, file_path):
    """
    Save a list of dictionaries to a JSON file.
    """
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=2)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")

def load_json(file_path):
    """
    Load a JSON file containing a list of dictionaries.
    """
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

################################################################################
#                        MAIN
################################################################################

docs_results = "docs/results/results_with_cells_llama3.2:latest_test_examples_1695_zero_shot_html.json"
all_claims = load_claims(docs_results)

model_name = "mistral:latest"
prompt_functions_with_categories = {
    "aggregation_prompt_generation": "aggregation",
    "negate_prompt_generation": "negate",
    "superlative_prompt_generation": "superlative",
    "comparative_prompt_generation": "comparative",
    "ordinal_prompt_generation": "ordinal",
    "unique_prompt_generation": "unique",
    "all_prompt_generation": "all",
    "none_prompt_generation": "none"
}

prompt_functions_with_our_categories = {
    "single_fact_verification_prompt_generation": "single_attribute_verification",
    "multi_fact_conjuction_prompt_generation": "multi_attribute_conjunction",
    "negation_absence_check_prompt_generation": "negation_absence_check",
    "numerical_threshold_check_prompt_generation": "numerical_threshold_check",
    "membership_association_prompt_generation": "membership_association",
    "aggregation_computation_prompt_generation": "aggregation_computation",
    "record_existence_prompt_generation": "record_existence",
    "comparative_analysis_prompt_generation": "comparative_analysis",
    "conditional_logic_prompt_generation": "conditional_logic"
}

prompt_functions_with_our_categories_bis = {
    "unclear_or_noisy_language_prompt_generation": "unclear_or_noisy_language",
    "numerical_reasoning_prompt_generation": "numerical_reasoning",
    "multistep_logic_prompt_generation": "multistep_logic",
    "negation_prompt_generation": "negation",
}

categorised_claims = error_analysis_pipeline(all_claims, model_name, prompt_functions_with_our_categories_bis)
save_to_json(categorised_claims, "PromptEngineering/Error_Analysis/our_category_claims_bis_mistral.json")

# for f in failed_predictions:
#     print(f['claim'])
#     print()

# def main():
#     parser = argparse.ArgumentParser(description="Run Error Analysis on a Result JSON File")
#     parser.add_argument("--result_file", type=str, required=True, help="Path to the result JSON file")
#     args = parser.parse_args()
#     analyze_errors(args.result_file)

# if __name__ == "__main__":
#     main()