# Llama Installation Guide

Below is a revised, step-by-step guide for installing and configuring Ollama on a Linux system:

1. Create an Installation Directory

    For example:

    ```bash
    mkdir -p ~/llama
    ```

2. Change to the Installation Directory

    ```bash
    cd ~/llama
    ```

3. Download the Ollama Tarball

    Use `curl` to download the Ollama binaries from the official source:

    ```bash
    curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
    ```

4. Extract the Downloaded Archive

    Decompress and extract the tarball to install the `ollama` binary and related files:

    ```bash
    tar -xzf ollama-linux-amd64.tgz
    ```

5. Verify the Extraction

    Check the current directory to ensure that the `bin` directory has been created:

    ```bash
    ls
    ```

6. Make the Binary Executable

    Grant execute permissions to the `ollama` binary:

    ```bash
    chmod +x ./bin/ollama
    ```

7. Add Ollama to Your PATH

    Replace `<path to installation directory>` with the actual path (e.g., `~/llama`):

    ```bash
    echo 'export PATH=<path to installation directory>/bin:$PATH' >> ~/.bashrc
    ```

8. Apply the Updated Environment Settings

    Reload your shell configuration to update environment variables:

    ```bash
    source ~/.bashrc
    ```

9. Test the Installation

    The output should include two warnings, confirming the installation:

    ```bash
    ollama -v
    ```

10. Set the Models Directory

    Specify where Ollama will store its models. Use the same installation directory path as above:

    ```bash
    echo 'export OLLAMA_MODELS=<path to installation directory>/models' >> ~/.bashrc
    source ~/.bashrc
    ```

11. Create the Models Directory

    Ensure the directory exists:

    ```bash
    mkdir -p $OLLAMA_MODELS
    ```

12. Start the Ollama Server

    Run the server in the background, allowing you to continue using the terminal:

    ```bash
    ollama serve &
    ```

13. Download the `llama3` Model

    Fetch and prepare the `llama3` model for use:

    ```bash
    ollama pull llama3
    ```

14. Download the `nomic-embed-text` Model

    Similarly, fetch the `nomic-embed-text` model:

    ```bash
    ollama pull nomic-embed-text
    ```

15. Navigate back to the Home Directory

    ```bash
    cd ~
    ```

16. Download the Miniconda Installer (Miniconda is a lightweight alternative to Anaconda, ideal for environments with limited disk space)

    Use `curl` to fetch the latest Miniconda installer for Linux:

    ```bash
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```

17. Run the Installer

    Launch the installation script:

    ```bash
    bash ~/Miniconda3-latest-Linux-x86_64.sh
    ```

    Follow the prompts to complete the installation, ensuring the Miniconda binary path is added to your `~/.bashrc` or `~/.zshrc`.

18. Initialize Conda

    If prompted during installation, initialize Conda. Otherwise, you can do this manually:

    ```bash
    conda init
    ```

    Restart your terminal to activate the changes.

19. Additional Cleanup

    To save disk space, clean up unused cache:

    ```bash
    conda clean --all
    ```

20. Create a Dedicated Environment

    Use Conda to set up an environment tailored to your project. For example:

    ```bash
    conda create -n llm python langchain langchain-core langchain-community gradio bs4 chromadb matplotlib spacy wikipedia-api tqdm ipykernel jupyter -c conda-forge
    ```

    Here, `-n llm` specifies the environment name as "llm," and the `-c conda-forge` flag ensures the use of the `conda-forge` channel.

21. Activate the Environment

    Enter the environment to start working with the installed packages:

    ```bash
    conda activate llm
    ```

22. Install ollama

    Use `pip` to install ollama as `conda`'s distribution seems to be currently broken

    ```bash
    pip install ollama
    ```

23. Fetch Spacy model

    ```bash
    python -m spacy download en_core_web_sm
    python -m spacy download en_core_web_md
    python -m spacy download en_core_web_trf
    ```

After these steps, Ollama and the specified models should be fully installed and ready to use.
