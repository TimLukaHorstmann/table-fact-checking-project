#!/bin/bash

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq is not installed. Please install jq to proceed."
    exit 1
fi

# Check if a file path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_json_file>"
    exit 1
fi

# Input file
input_file="$1"

# Output file
output_file="$(dirname "$input_file")/stats_$(basename "$input_file" .json).json"

# Total number of claims in the original file
total_original_claims=12279

# Initialize counters
true_positives=0
false_positives=0
true_negatives=0
false_negatives=0
object_count=0

# Iterate over each object in the JSON file
while IFS= read -r obj; do
    predicted=$(echo "$obj" | jq -r '.predicted_response')
    true=$(echo "$obj" | jq -r '.true_response')

    if [ "$predicted" -eq 1 ] && [ "$true" -eq 1 ]; then
        ((true_positives++))
    elif [ "$predicted" -eq 1 ] && [ "$true" -eq 0 ]; then
        ((false_positives++))
    elif [ "$predicted" -eq 0 ] && [ "$true" -eq 0 ]; then
        ((true_negatives++))
    elif [ "$predicted" -eq 0 ] && [ "$true" -eq 1 ]; then
        ((false_negatives++))
    fi

    # Increment object count
    ((object_count++))

    # Print progress every 200 objects
    if (( object_count % 500 == 0 )); then
        echo "Processed $object_count objects..."
    fi
done < <(jq -c '.[]' "$input_file")

# Calculate precision and recall
precision=$(echo "scale=4; $true_positives / ($true_positives + $false_positives)" | bc)
recall=$(echo "scale=4; $true_positives / ($true_positives + $false_negatives)" | bc)

# Calculate the ratio of correct predictions to total claims
correct_predictions=$((true_positives + true_negatives))
correct_predictions_ratio=$(echo "scale=4; $correct_predictions / $total_original_claims" | bc)

# Create a JSON object with the statistics
stats=$(jq -n \
    --argjson total_processed_claims "$((true_positives + false_positives + true_negatives + false_negatives))" \
    --argjson correct_predictions "$correct_predictions" \
    --argjson precision "$precision" \
    --argjson recall "$recall" \
    --argjson correct_predictions_ratio "$correct_predictions_ratio" \
    '{
        "processed claims (out of 12279 total) (the remainder is generated code containing errors and so not ran)": $total_processed_claims,
        "correct predictions made": $correct_predictions,
        "precision on those processed claims": $precision,
        "recall on those processed claims": $recall,
        "correct_predictions/12279_total_claims": $correct_predictions_ratio
    }')

# Output the statistics to a file
echo "$stats" | jq . > "$output_file"

echo "Statistics have been saved to $output_file"

