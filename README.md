# iViz Modular RAG

**Overview**

This project implements a flexible chatbot system that utilizes Retrieval-Augmented Generation (RAG) models and the OpenAI API for natural language interactions. Key features include:

* **Multiple RAG Models:** The system supports chaining multiple RAG models, each specialized in different domains or styles of communication. This allows for more comprehensive and nuanced responses.
* **Customizable Character:** The chatbot's personality and knowledge base can be modified by providing different character and model initialization data (`.json` files).
* **OpenAI Integration:** Leverages the powerful OpenAI API for text generation, ensuring the chatbot can engage in diverse conversations.

**Installation**

The code was developed in Python 3.11.7. 

1. **Create a virtual environment:** 
   ```bash
   python3 -m venv env  # Create environment named 'env'
   source env/bin/activate  # Activate the environment (Linux/macOS)
   env\Scripts\activate  # Activate the environment (Windows)

2. **Install requirements:** 
   ```bash
   pip install -r requirements.txt

**Class Definitions**

* **OpenAIChatbot:** Interacts with the OpenAI API for chatbot functionality.
* **Embedder:** Encodes text into embeddings and allows efficient similarity search.
* **DataProcessor:** Through `split_text()`, it splits a text file into a numpy array of substrings based on a separator or chunk length.
* **RAG:** Represents a Retrieval-Augmented Generation (RAG) model.
* **RAGChain:** Utilizes multiple RAG models for question answering, combining their strengths.
