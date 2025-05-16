#!/bin/bash

# === Configuration ===

# Path to the Python augmentation script
PYTHON_SCRIPT="gpt_augment.py"

# Directory containing the original dataset JSON files
# Assumes filenames like ANLI.json, ARC.json, etc.
INPUT_DIR="./data/training_data"

# Directory where the augmented JSON files will be saved
OUTPUT_DIR="./data/gpt/"

# List of dataset tasks to process
# Ensure these names match the expected --task argument and the input filenames (without .json)
TASKS=("ANLI" "ARC" "CSQA" "Date" "GSM8K" "MATH" "SQA")

# OpenAI model to use for augmentation (e.g., "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
MODEL="gpt-3.5-turbo"

# Number of answer augmentations to generate per question (set to 0 to disable)
K_ANSWER_AUG=3

# === Script Execution ===

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipelines return the exit status of the last command to exit with a non-zero status,
# or zero if no command exited with a non-zero status.
set -o pipefail

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found in the current directory."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Output directory '$OUTPUT_DIR' ensured."
echo "Using model: $MODEL"
echo "Answer augmentations per question (k): $K_ANSWER_AUG"

# Loop through each task and run the Python script
for TASK in "${TASKS[@]}"; do
    INPUT_FILE="${INPUT_DIR}/${TASK}.json"
    OUTPUT_FILE="${OUTPUT_DIR}/${TASK}_augmented.json"

    echo "--------------------------------------------------"
    echo "Processing Task: $TASK"
    echo "Input:  $INPUT_FILE"
    echo "Output: $OUTPUT_FILE"
    echo "--------------------------------------------------"

    # Check if the specific input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: Input file '$INPUT_FILE' not found for task '$TASK'. Skipping."
        continue # Move to the next task in the loop
    fi

    # Construct the command arguments
    # Using an array handles potential spaces in filenames/paths safely
    COMMAND_ARGS=(
        python "$PYTHON_SCRIPT"
        --input-file "$INPUT_FILE"
        --output-file "$OUTPUT_FILE"
        --task "$TASK"
        --model "$MODEL"
        --k-answer-aug "$K_ANSWER_AUG"
        # Add other arguments like --start-index or --max-items if needed
        # e.g., --max-items 100
    )

    # Execute the command
    echo "Executing: ${COMMAND_ARGS[*]}" # Print the command being run
    if "${COMMAND_ARGS[@]}"; then
        echo "Successfully completed task: $TASK"
    else
        # Because of 'set -e', the script will exit here if the python command fails.
        # If you want the script to continue processing other tasks even if one fails,
        # remove 'set -e' from the top and add more specific error handling here.
        echo "Error occurred during task: $TASK. Script halted."
        exit 1 # Explicitly exit, though set -e should handle this
    fi

done

echo "=================================================="
echo "All specified tasks processed successfully."
echo "Augmented files saved in: $OUTPUT_DIR"
echo "=================================================="

exit 0
