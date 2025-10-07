# Optimizing Multi-Hop Reasoning in RAG Systems: An Agentic Graph RAG Framework with Efficient Reranking

## 🚀 **Installation**
```bash
cd AGRAG
pip install -e .
```
## ⚙️ **Quick Start**
**1. Setup API Keys**
* Configure your API keys in the environment and in the `llm.py` file.

**2. Insert Data and Query**

To experiment with your own data:
* Place your text file (e.g., text.txt) in the project directory.
* Run the framework:
```bash
python run.py
```

This script:
* Inserts your file as corpus data.
* Executes queries using the pipeline.
* Prints the final reasoning-based answer.

**Working with Multiple Files**
* You can also test the framework on a corpus of multiple files by modifying the insertion section in `run.py` to load several documents sequentially.