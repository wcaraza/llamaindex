# llamaindex
llamaindex_poc

### Prerequisites:

Operating system: Linux, macOS or Windows
Python 3.7 or higher
Access to terminal or command console

## Llama Index installation:

1. Update the package manager:

pip3 install --upgrade pip

2. Create a virtual environment:

python -m venv llamaindex_env
source llamaindex_env/bin/activate

3. Install LlamaIndex:

pip3 install llama-index

4. Check the installation:

python -c "import llama_index; print(llama_index.__version__)"

5. Install dependencies:

pip3 install -r requirements.txt


## Ollama installation:

For Linux:
pip3 install --upgrade pip

For Mac, download from the following url:

https://github.com/ollama/ollama

2. Check the installation:

% ollama

3. Install pre-trained models:

% ollama run mistral                                            

