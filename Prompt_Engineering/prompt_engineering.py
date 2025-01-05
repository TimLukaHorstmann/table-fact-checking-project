
# from langchain.llms import Ollama
import json
import pandas as pd
import os

#### Preprocessing #####
# def remove_answers_from_file(infile):
#     with open(infile, "r") as file:
#         full_cleaned = json.load(file)
#     full_cleaned_no_answers = {}
#     for key, value in full_cleaned.items():
#         new_value = [value[0], None, value[2], value[3]]
#         full_cleaned_no_answers[key] = new_value
#     with open("full_cleaned_no_answers.json", "w") as file:
#         json.dump(full_cleaned_no_answers, file, indent=2)
#     print("Processed file saved as full_cleaned_no_answers.json")

print("current directory:", os.getcwd())

full_cleaned_path = os.getcwd()+"/full_cleaned.json"
all_csv_folder = os.getcwd()+"/all_csv/"

one_shot_example = """
        Example: 

        | Helicopter                | Description                  | Max Gross Weight| Total Disk Area|
        |---------------------------|------------------------------|-----------------|----------------|
        | Lockheed Martin X34       | Light Utility Helicopter     | 26000 lb        | 1205 ft square |
        | Jambo Force 54            | Turboshaft Utility Helicopter| 3200 lb         | 435 ft square  | 
        | Chinook AC432             | Tandem Rotor Helicopter      | 12500 lb        | 3453 ft square | 
        | AOL 11-b                  | Massive Helicopter           | 143000 lb       | 9456 ft square | 
        | Super flight 600          | Heavy-Lift Helicopter        | 3900 lb         | 2890 ft square |

        Claim: "The Jambo Force 54 be the aircraft with the lowest max gross weight."

        Answer: ENTAILED as we can see from the table, the "Max Gross Weight" of the Jambo Force 54 is 3200 lb, while all the other helicopters have a max gross weight which is >= to 3200 lb.
        Therefore the Jambo Force 54 has the lowest value for Max Gross Weight, confirming the claim. Hence, the answer is ENTAILED.
        """
few_shot_example = """
        Example 1:

        | Country       | Capital      | Population (millions) | Continent      |
        |---------------|--------------|-----------------------|----------------|
        | USA           | Washington   | 331                   | North America  |
        | Canada        | Ottawa       | 38                    | North America  |
        | Australia     | Canberra     | 25                    | Oceania        |

        
        Claim 1: "Canada has a higher population than Australia."

        Answer 1: ENTAILED, because from the table, Canada has a population of 38 million, while Australia has 25 million.
        Since 38 > 25, Canada does have a higher population, so the claim is accurate. Hence, the answer is ENTAILED.


        Claim 2: "There be two countries in the continent of Oceania."

        Answer 2: REFUSED, because as we can see from the table, only one country (Australia) is in the continent "Oceania".
        Since one is not equal to two, this claim is therefore false. Hence, the answer is REFUSED.

        #########
        Example 2:

        | Brand          | Category      | Price (USD) | Stock Count |
        |----------------|---------------|-------------|-------------|
        | Nike           | Shoes         | 120         | 50          |
        | Adidas         | Shoes         | 100         | 30          |
        | Puma           | Sweatpants    | 90          | 60          |

        Claim 1: "Nike shoes are the cheapest option available in stock."

        Answer 1: REFUTED, because from the table, Nike's shoes cost $120, Adidas's cost $100 (we don't consider Puma's sweatpants since they are not in the "Shoes" category!).
        Adidas's shoes are the cheapest shoes, not Nike's. Therefore, the claim is incorrect. Hence, the answer is REFUTED.


        Claim 2: "Puma's sweatpants are the most expensive sweatpants."
        
        Answer 2: ENTAILED. Since the only Sweatpants are the ones of the Puma brand, they are the most expensive (also technically the cheapest at the same time).
        Therefore the claim is correct. Hence, the answer is ENTAILED.


        #########
        Example 3:

        | City        | Average Temperature (°C)  | Rainfall (mm/year) | Country     |
        |-------------|---------------------------|--------------------|-------------|
        | Tokyo       | 16.1                      | 1520               | Japan       |
        | Cairo       | 22.8                      | 20                 | Egypt       |
        | London      | 11.6                      | 760                | UK          |

        Claim 1: "Cairo receives the most rainfall among the listed cities."

        Answer 1: REFUTED, because Cairo has 20 mm/year of rainfall, while Tokyo has 1520 mm/year, and London has 760 mm/year.
        Tokyo receives the most rainfall, not Cairo. Hence, the claim is REFUTED.


        Claim 2: "There are two cities that have a rainfall of over 500 and that also have an average temperature of over 16°C."

        Answer 2: REFUTED, because here, the cities having a rainfall of over 500 are Tokyo (since Tokyo's rainfall, 1520, is over 500), and London (since London's rainfall, 760, is over 500), which makes two cities, 
        while the cities that have an average temperature of over 16°C are Cairo (22.8 >= 16) and Tokyo (16.1 >= 16), which also makes two cities, but these two cities (Cairo, Tokyo) are different from the two cities that satisfied the rainfall condition (Tokyo, London).
        There is only one city that satisfies both conditions, that is, Tokyo. Therefore, there are not two cities that satisfy the two conditions of the claim. Therefore, the claim is false, and the answer is REFUSED.
        """



def generate_prompt(table_id, claim, learning_type="zero_shot"):
    """
    Generate a prompt for a specific table and statement.
    
    Args:
        table_id (str): The table filename from full_cleaned.json.
        statement (str): The statement (claim) to be validated.
        learning_type (str): Type of prompt engineering ("zero_shot", "one_shot", "few_shot").
    
    Returns:
        str: A formatted prompt string.
    """
    # Load the corresponding table as a DataFrame
    table_path = f"{all_csv_folder}/{table_id}"
    table = pd.read_csv(table_path, delimiter="#")  # Assuming '#' is the delimiter in the CSV

    # Format the table for the prompt
    table_formatted = table.to_markdown(index=False)  # Converts table to Markdown for easy display
    if learning_type == "zero_shot":
        prompt = f"""
        Table:
        {table_formatted}

        Claim: "{claim}"

        Does the table support the claim? Answer "ENTAILED" (if the claim is true) or "REFUTED" (if the claim is false). 
        Make sure to check each mentioned entity and relationships given by the claim, before giving your answer. 
        """
    elif learning_type == "one_shot":
        prompt = f"""
        Use the example given below to get an idea of the expected task.

        {one_shot_example}

        Now that you have reviewed the example, you will proceed to a similar task.

        Table:
        {table_formatted}

        Claim: "{claim}"

        Does the table support the claim? Answer "ENTAILED" (if the claim is true) or "REFUTED" (if the claim is false). 
        Make sure to check each mentioned entity and relationships given by the claim, before giving your answer.
        """

    elif learning_type == "few_shot":
        prompt = f"""
        Use the examples given below to get an idea of the expected task.

        {few_shot_example}

        Now that you have reviewed the examples, you will proceed to a similar task.
        
        Table:
        {table_formatted}

        Claim: "{claim}"

        Does the table support the claim? Answer "ENTAILED" (if the claim is true) or "REFUTED" (if the claim is false). 
        Make sure to check each mentioned entity and relationships given by the claim, before giving your answer.
        """
        

    return prompt.strip()





def test_model_on_claims(model, full_cleaned_data, test_all=False, N=10, learning_type="zero_shot"):
    """
    Test the model on the first N claims or all claims from full_cleaned.json.

    Inputs:
        model: LLM model instance.
        full_cleaned_data (dict): Data from full_cleaned.json.
        test_all (bool): Whether to test all claims.
        N (int): Number of claims to test if test_all is False.
        learning_type (str): Type of prompt engineering ("zero_shot", "one_shot", "few_shot").
    """
    results = []
    keys = list(full_cleaned_data.keys())
    limit = len(keys) if test_all else min(N, len(keys))

    for i in range(limit):
        table_id = keys[i]
        claims = full_cleaned_data[table_id][0]
        
        for claim in claims:
            prompt = generate_prompt(table_id, claim, learning_type)
            response = model.predict(prompt)
            results.append({"table_id": table_id, "claim": claim, "response": response})

    return results





def calculate_accuracy(results, full_cleaned_data):
    """
    Calculate the accuracy of the model based on its predictions.

    Args:
        results (list): List of dictionaries with table_id, claim, and response.
        full_cleaned_data (dict): Original data from full_cleaned.json containing ground truth labels.

    Returns:
        float: Accuracy as a percentage.
    """
    correct = 0
    total = len(results)

    for result in results:
        table_id = result["table_id"]
        claim = result["claim"]
        predicted = result["response"].strip().upper()

        # Locate the corresponding label in full_cleaned_data
        claims = full_cleaned_data[table_id][0]
        labels = full_cleaned_data[table_id][1]

        if claim in claims:
            idx = claims.index(claim)
            true_label = "ENTAILED" if labels[idx] == 1 else "REFUTED"

            if predicted == true_label:
                correct += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy



# Load full_cleaned.json
with open(full_cleaned_path, "r") as f:
    full_cleaned_data = json.load(f)

# # Example: Select a table and claim
# table_id = "1-10006830-1.html.csv"  # Example table ID
# for table = 
# first10claims = full_cleaned_data[table_id][0]  # Get list of claims
# labels = full_cleaned_data[table_id][1]      # Get corresponding labels

# statement = statements[0]  # Example claim
# prompt = generate_prompt(table_id, claim=, )

# # Output the test prompt
# print(prompt)


llm = Ollama(model = "llama3.2")

# Parameters
test_all = False
N = 10  # Number of claims to test if test_all is False
learning_type = "zero_shot"

# Run the testing function
results = test_model_on_claims(llm, full_cleaned_data, test_all=test_all, N=N, learning_type=learning_type)

# Print results
for result in results:
    print(f"Table ID: {result['table_id']}\nClaim: {result['claim']}\nResponse: {result['response']}\n")



accuracy = calculate_accuracy(results, full_cleaned_data)
print(f"Accuracy: {accuracy:.2f}%")