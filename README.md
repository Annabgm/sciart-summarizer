# sciart-summarizer
Scientific Article Summarizer

## Overview
`sciart-summarizer` is a Streamlit-based application designed to summarize scientific articles. It leverages LangChain, OpenAI models, and Chroma for efficient document processing, summorization, and citation management. The application allows users to upload PDF files and receive concise summaries with citation information. Additionally, it tracks API usage costs for transparency.

---

## Features
- **PDF Upload**: Upload one or multiple PDF files for processing.
- **Summarization**: Generate summaries of scientific articles with proper citations.
- **Citation Management**: Extract and format citations from the processed documents.
- **Spending Tracking**: Monitor API usage costs, including token usage and total cost.

---

## Technical details
- The solution uses `LangChain` as the main framework for interacting with the OpenAI LLM model. 
- The `citeproc` library is responsible for formatting citations. Additional citation styles can be added as needed.
- For greater flexibility, runtime configuration is used in the graph to control the length of the summary.
- The adjusted RAG (Retrieval-Augmented Generation) logic is employed to accomplish the summarization task. Uploaded files are split into chunks and stored in a vector database. When the summarization topic is defined, the required number of chunks is retrieved and used as context for the LLM request.

---

## Future plan
The final implementation involves two additional steps:
- Creating a Docker Compose setup with three containers: the main module, a vector database, and LangSmith.
- Adding a new tab for LLM tracking using LangSmith to optimize prompts.

`Note`: 
The current implementation uses the Chroma vector database, which relies on SQLite. Chroma offers better filtering flexibility compared to FAISS. However, exploring a solution backed by MongoDB could provide additional scalability and robustness.

Sometimes, the LLM response does not exactly match the desired output format. This can be improved by providing examples and incorporating human feedback for fine-tuning.

---

## Installation

### Prerequisites
- Python 3.11 or higher
- Virtual environment (optional but recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sciart-summarizer.git
   cd sciart-summarizer
   ```

2. Install dependencies:
`pip install -r requirements.txt`

3. Set up OpenAI API keys:
`export OPENAI_API_KEY="your_openai_api_key"`

4. Run the application:
`streamlit run src/app.py`
