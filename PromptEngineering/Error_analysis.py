def load_predictions(root_dir: str, pattern: str = "results_with_cells_*.json") -> list:
    """
    Scan the given root_dir (and its subdirectories) for files matching the pattern "results_with_cells_*.json",
    and return a list of dictionaries for the predictions where the model failed (i.e., predicted_response != true_response).
    """
    failed_predictions = []
    successful_predictions = []

    # Use glob to find all matching files, recursively in subdirectories
    print("Searching in directory:", os.path.abspath(root_dir))
    file_list = glob.glob(os.path.join(root_dir, '**', pattern), recursive=True)
    if not file_list:
        print("No intermediate results files found matching pattern.")
        return failed_predictions, successful_predictions
    
    # Initialize a counter for the IDs
    id_counter = 1

    for file_path in tqdm(file_list):
        try:
            with open(file_path, "r") as f:
                results = json.load(f)
        except Exception as e:
            print(f"Could not load {file_path}: {e}")
            continue

        for result in results:
            # Add an ID to each claim
            result["ID"] = id_counter
            id_counter += 1  # Increment the ID counter

            # Check if the model's predicted response is different from the true response
            if result.get("predicted_response") != result.get("true_response"):
                # Create a dictionary with relevant details for failed predictions
                failed_prediction = {
                    "ID": result["ID"],
                    "predicted_response": result["predicted_response"],
                    "true_response": result["true_response"],
                    "claim": result["claim"]
                }
                failed_predictions.append(failed_prediction)
            else:
                # Create a dictionary with relevant details for successful predictions
                successful_prediction = {
                    "ID": result["ID"],
                    "predicted_response": result["predicted_response"],
                    "true_response": result["true_response"],
                    "claim": result["claim"]
                }
                successful_predictions.append(successful_prediction)
    
    return failed_predictions, successful_predictions
