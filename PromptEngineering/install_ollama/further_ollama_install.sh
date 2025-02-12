#!/bin/bash
# Exit immediately if a command exits with a non-zero status.

# NOTE: This script assumes that the relevant environment variables are already set in .bashrc.
# Please look at the Llama installation guide for more information on setting these variables.
# This script can be used once the full installation has been done once.

set -e

# Define variables
BASE_DIR="/Data/tlh45"
LLAMA_DIR="$BASE_DIR/llama"
OLLAMA_TGZ="ollama-linux-amd64.tgz"
OLLAMA_URL="https://ollama.com/download/ollama-linux-amd64.tgz"

# Create necessary directories
mkdir -p "$LLAMA_DIR"
cd "$LLAMA_DIR"

# Download the ollama tarball
curl -L "$OLLAMA_URL" -o "$OLLAMA_TGZ"

# Extract the tarball
tar -xzf "$OLLAMA_TGZ"

# Make the ollama binary executable
chmod +x ./bin/ollama

# Create the models directory
mkdir -p "$OLLAMA_MODELS"

# Start the ollama server in the background
./bin/ollama serve &

# Give the server a moment to start up before pulling models.
sleep 5

# Pull the specified models
./bin/ollama pull deepseek-r1:32b
./bin/ollama pull phi4:latest
./bin/ollama pull llama3.2:latest
./bin/ollama pull mistral:latest

echo "All steps completed successfully."
