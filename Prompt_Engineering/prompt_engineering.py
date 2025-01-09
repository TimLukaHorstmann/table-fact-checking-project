from langchain_ollama import OllamaLLM
import json
import pandas as pd
import os
from tqdm import tqdm 
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
import numpy as np
import pickle


#### Preprocessing #####
# Merging function to combine r1 and r2 data
def merge_training_data(file1, file2, output_file):
    with open(file1, 'r') as f:
        results1 = json.load(f)

    with open(file2, 'r') as f:
        results2 = json.load(f)

    results2.update(results1)

    with open(output_file, 'w') as f:
        json.dump(results2, f, indent=2)

    print(f"Successfully merged {file1} and {file2} into {output_file}.")


#Initial data
file_simple_r1 = "r1_training_all.json" # file with simple claims
file_complex_r2 = "r2_training_all.json" # file with complex claims
all_csv_folder = os.getcwd()+"/all_csv/" # folder with the tables themselves
test_set = "test_examples.json"
val_set = "val_examples.json"
# output_file = "full_claim_file.json"
# merge_training_data(file_simple_r1, file_complex_r2, output_file)
# with open(output_file, "r") as f:
#     claim_file = json.load(f) # file containing the claims for each table

with open(file_simple_r1, 'r') as f:
    r1_simple = json.load(f)
    print(f.readline())
with open(file_complex_r2, 'r') as f:
    r2_complex = json.load(f)
with open(file_complex_r2, 'r') as f:
    test_json = json.load(f)
with open(file_complex_r2, 'r') as f:
    val_json = json.load(f)

print("current directory:", os.getcwd())

######## Create examples for one_shot and few_shot learnings:
one_shot_example = """
        Example: 

        | Helicopter                | Description                  | Max Gross Weight| Total Disk Area|
        |---------------------------|------------------------------|-----------------|----------------|
        | Lockheed Martin X34       | Light Utility Helicopter     | 26000 lb        | 1205 ft square |
        | Jambo Force 54            | Turboshaft Utility Helicopter| 3200 lb         | 435 ft square  | 
        | Chinook AC432             | Tandem Rotor Helicopter      | 12500 lb        | 3453 ft square | 
        | AOL 11-b                  | Massive Helicopter           | 143000 lb       | 9456 ft square | 
        | Super flight 600          | Heavy-Lift Helicopter        | 3900 lb         | 2890 ft square |

        Claim: "The Super flight 600 is the aircraft with the lowest max gross weight of the aircrafts with total disk area above 2000 ft square."

        Answer: TRUE. We can see from the table, that only three helicopters have a total disk area above 2000 ft square (Super flight 600, Chinook AC432,  AOL 11-b), and among these three, the Super flight 600 is the one with the lowest max gross weight, confirming the claim.
        """


few_shot_example = """
        Example 1:

        | Country       | Capital      | Population (millions) | Continent      |
        |---------------|--------------|-----------------------|----------------|
        | USA           | Washington   | 331                   | North America  |
        | Canada        | Ottawa       | 38                    | North America  |
        | Australia     | Canberra     | 25                    | Oceania        |

        
        Claim 1: "Canada has a higher population than Australia."

        Answer 1: TRUE. Because from the table, Canada has a population of 38 million, which is greater than Australia's population of 25 million.


        Claim 2: "There are two countries in the continent of Oceania."

        Answer 2: FALSE, because as we can see from the table, only one country (Australia) is in the continent "Oceania", not two.

        #########
        Example 2:

        | Brand          | Category      | Price (USD) | Stock Count |
        |----------------|---------------|-------------|-------------|
        | Nike           | Shoes         | 120         | 50          |
        | Adidas         | Shoes         | 100         | 30          |
        | Puma           | Sweatpants    | 90          | 60          |

        Claim 1: "Nike shoes are the cheapest option available in stock."

        Answer 1: FALSE, because from the table, Nike's shoes cost $120, Adidas's cost $100 (we don't consider Puma's sweatpants since they are not in the "Shoes" category), and so the Adidas shoes are the cheapest shoes, not the Nike shoes.


        Claim 2: "Puma's sweatpants are the most expensive sweatpants."
        
        Answer 2: TRUE. Since the only Sweatpants are the ones of the Puma brand, they are the most expensive (also technically the cheapest at the same time).

        #########
        Example 3:

        | City        | Average Temperature (°C)  | Rainfall (mm/year) | Country     |
        |-------------|---------------------------|--------------------|-------------|
        | Tokyo       | 16.1                      | 1520               | Japan       |
        | Cairo       | 22.8                      | 20                 | Egypt       |
        | London      | 11.6                      | 760                | UK          |

        Claim 1: "Cairo receives the most rainfall among the listed cities."

        Answer 1: FALSE, because Cairo has 20 mm/year of rainfall, while Tokyo has 1520 mm/year, and London has 760 mm/year.
        Tokyo receives the most rainfall, not Cairo. Hence, the claim is FALSE.


        Claim 2: "There are two cities that have a rainfall of over 500 and that also have an average temperature of over 16°C."

        Answer 2: FALSE, because here, the cities having a rainfall of over 500 are Tokyo (since Tokyo's rainfall, 1520, is over 500), and London (since London's rainfall, 760, is over 500), which makes two cities, 
        while the cities that have an average temperature of over 16°C are Cairo (22.8 >= 16) and Tokyo (16.1 >= 16), which also makes two cities, but these two cities (Cairo, Tokyo) are different from the two cities that satisfied the rainfall condition (Tokyo, London).
        There is only one city that satisfies both conditions, that is, Tokyo. Therefore, there are not two cities that satisfy the two conditions of the claim. Therefore, the claim is FALSE, and the answer is FALSE.
        """

end_of_prompt = """
    "Answer with one of the following options:

    **TRUE**: If the claim is true (the table supports it).
    **FALSE**: If the claim is false (the table does not support it).
        
    The claim may involve multiple conditions. Please consider each condition in the claim and validate them against the table's data.
    Please respond only with either TRUE or FALSE (capitalized!), without any extra text.
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
    table_path = f"{all_csv_folder}/{table_id}"
    table = pd.read_csv(table_path, delimiter="#")

    # Format table for the prompt
    table_formatted = table.to_markdown(index=False)  # Converts table to Markdown for easy display
    if learning_type == "zero_shot":
        prompt = f"""
        Table:
        {table_formatted}

        Claim: "{claim}"

        {end_of_prompt}
        """
    elif learning_type == "one_shot":
        prompt = f"""
        Use the example given below to get an idea of the task (claim verification using a table).

        {one_shot_example}

        Now that you have reviewed the example, proceed to the following task.

        Table:
        {table_formatted}

        Claim: "{claim}"

        {end_of_prompt}
        """

    elif learning_type == "few_shot":
        prompt = f"""
        Use the examples given below to get an idea of the task (claim verification using a table).

        {few_shot_example}

        Now that you have reviewed the examples, proceed to the following task.
        
        Table:
        {table_formatted}

        Claim: "{claim}"

        {end_of_prompt}
        """
    return prompt.strip()


def test_model_on_claims(model, 
                         full_cleaned_data, 
                         test_all=False, 
                         N=10, 
                         learning_type="zero_shot"):
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
    for i in tqdm(range(limit)):
        table_id = keys[i]
        claims = full_cleaned_data[table_id][0]
        labels = full_cleaned_data[table_id][1] 

        for idx, claim in enumerate(claims):
            prompt = generate_prompt(table_id, claim, learning_type)
            response = model.invoke(prompt).strip()
            if "TRUE" in response or "**TRUE**" in response:
                predicted_label = 1
            else:
                predicted_label = 0

            true_label = labels[idx]

            results.append({
                "table_id": table_id,
                "claim": claim,
                "predicted_response": predicted_label,
                "resp": response,
                "true_response": true_label
            })
    return results

#### Plotting and results
def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    """
    Plot confusion matrix using Seaborn heatmap.
    
    Args:
        y_true: True labels (list or array).
        y_pred: Predicted labels (list or array).
        classes: List of class labels.
        title: Title of the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plot ROC curve.
    
    Args:
        fpr: False Positive Rate.
        tpr: True Positive Rate.
        roc_auc: Area under the ROC curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


import pickle

def calculate_and_plot_metrics(results, 
                               save_dir="results_plots", 
                               save_stats_file="summary_stats.pkl", 
                               learning_type="", 
                               dataset_type=""):
    """
    Calculate precision, recall, F1-score, accuracy, TP, FP, TN, FN.
    Plot confusion matrix and ROC curve,
    Save all to files.

    Args:
        results: List of dictionaries containing "true_response" and "predicted_response".
        save_dir: Directory to save plots.
        save_stats_file: File path to save stats in pickle format.
        learning_type: Type of learning method used ("zero_shot", "one_shot", "few_shot").
        dataset_type: Type of dataset ("Simple", "Complex").
    """
    y_true = [result['true_response'] for result in results]
    y_pred = [result['predicted_response'] for result in results]

    classes = [0, 1] 

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):  # normal behavior
        tp, fn, fp, tn = cm.ravel()
    else:  # add 0 manually for empty classes
        tp = cm[1, 1] if cm.shape[0] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        fp = cm[0, 1] if cm.shape[0] > 1 else 0
        tn = cm[0, 0] if cm.shape[0] > 1 else 0

    # Save confusion matrix and ROC curve as plots
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {learning_type} Learning - {dataset_type} Dataset')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix_{learning_type}_{dataset_type}.png")
    plt.close()  # Close to avoid overlap in the next plot

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {learning_type} Learning - {dataset_type} Dataset')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curve_{learning_type}_{dataset_type}.png")
    plt.close()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Save statistics to a dictionary
    stats = {
        "learning_type": learning_type,
        "dataset_type": dataset_type,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),  # Save as list (can also keep it as numpy array if needed)
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn
    }

    # Save stats dictionary to pickle file
    with open(f"{save_dir}/{save_stats_file}", "wb") as f:
        pickle.dump(stats, f)

    print(f"{learning_type} Learning - {dataset_type} Dataset:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")


######################

### Running model
# used models: llama3.2 , mistral
# future use: qwq? large but possible

# Parameters
test_all = True
N = 3  # Number of claims to test (if test_all is False)

summary_metrics = {}
for model in ["mistral", "llama3.2"]:

    llm = OllamaLLM(model=model)
    for type_learning in ["few_shot"]:

        #for running on test/val sets only: 
        datasets = [{"test_set":test_json}, {"val_set":val_json}]
        
        # for running on simple or complex dataset only:
        # datasets = [{"simple_set": r1_simple}, {"complex_set":r2_complex}]

        for index, dataset in enumerate(datasets):
            dataset_type = f"{list(datasets[index].keys())[0]}"
            results = test_model_on_claims(llm, dataset[dataset_type], test_all=test_all, N=N, learning_type=type_learning)
            saving_directory = f"results_plots_{model}_{dataset_type}_{"all" if test_all else N}"
            if not os.path.exists(saving_directory):
                os.makedirs(saving_directory)

            # save accuracies, plots in appropriate folder
            calculate_and_plot_metrics(results, 
                                    save_dir=saving_directory,  
                                    learning_type=type_learning, 
                                    dataset_type=dataset_type
                                    )