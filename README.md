# AI PDF QA Bot

An intelligent PDF analysis and question-answering system built with Streamlit, LangChain, and Groq. This application allows users to upload PDF documents, get summaries, visualize document structure, and ask questions about the content.

## Features

- 📄 PDF Document Analysis
- 🤖 AI-powered Question Answering
- 📊 Document Structure Visualization
- 📝 Automatic Summary Generation
- 🔍 Semantic Search Capabilities

## Prerequisites

- Python 3.8+
- Groq API Key
- Graphviz (for flowchart visualization)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-PDF-QA-Bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Graphviz:
- Windows: Download and install from [Graphviz website](https://graphviz.org/download/)
- Linux: `sudo apt-get install graphviz`
- Mac: `brew install graphviz`

4. Create a `.env` file in the project root and add your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload a PDF document using the sidebar

4. Wait for the document to be processed (this may take a few moments)

5. Explore the following features:
   - View the document summary
   - See the document structure visualization
   - Ask questions about the document content

## Technical Details

The application uses several key technologies:

- **Streamlit**: For the web interface
- **LangChain**: For the AI pipeline
- **Groq**: For the language model
- **Qdrant**: For vector storage
- **HuggingFace Embeddings**: For text embeddings
- **Graphviz**: For flowchart visualization

## Supported LLM Models

The application supports multiple Groq models:
- llama3-8b-8192
- llama3-70b-8192
- mixtral-8x7b-32768
- gemma-7b-it

## Troubleshooting

If you encounter the "no running event loop" error, run the application with:
```bash
streamlit run app.py --server.fileWatcherType none
```

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://streamlit.io/)
- [Groq](https://groq.com/)
- [HuggingFace](https://huggingface.co/)