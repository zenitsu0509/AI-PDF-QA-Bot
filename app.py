import streamlit as st
import os
import tempfile
import graphviz
import io
import base64
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Qdrant client
import qdrant_client

@st.cache_resource(show_spinner="Loading PDF...")
def load_and_split_pdf(pdf_file):
    """Loads PDF, extracts text, and splits it into chunks."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load() # Returns list of Document objects
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents) # Split Documents directly
        full_text = " ".join([doc.page_content for doc in documents]) # Get full text for summary/flowchart
        os.unlink(tmp_file_path) # Clean up temporary file
        return texts, full_text
    except Exception as e:
        st.error(f"Error loading or splitting PDF: {e}")
        return None, None

@st.cache_resource(show_spinner="Creating Vector Store...")
def create_vector_store(_texts, _embeddings):
    """Creates an in-memory Qdrant vector store."""
    try:
        # Use in-memory Qdrant client
        client = qdrant_client.QdrantClient(":memory:")
        qdrant = Qdrant.from_documents(
            _texts,
            _embeddings,
            location=":memory:",  # Specify in-memory storage
            collection_name="pdf_chunks"
        )
        return qdrant
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

@st.cache_data(show_spinner="Generating Summary...")
def get_summary(_llm, _full_text):
    """Generates summary using the LLM."""
    prompt_template = """
    Provide a concise summary of the following text:
    ---
    {text}
    ---
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | _llm # Use LCEL (LangChain Expression Language)
    try:
        summary = chain.invoke({"text": _full_text})
        return summary.content # Extract content from AIMessage
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Could not generate summary."

@st.cache_data(show_spinner="Generating Flowchart Code...")
def get_flowchart_dot(_llm, _full_text):
    """Generates flowchart in DOT format using the LLM."""
    prompt_template = """
    Analyze the following text and generate a flowchart representation of its core process, structure, or key steps in Graphviz DOT format.
    Focus on the main elements and their relationships.
    Only output the valid DOT code. Do not include explanations or markdown formatting like ```dot ... ```.
    If no clear process or structure suitable for a flowchart is found, output 'NO_FLOWCHART'.
    ---
    TEXT:
    {text}
    ---
    DOT CODE:"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | _llm
    try:
        dot_code_response = chain.invoke({"text": _full_text})
        dot_code = dot_code_response.content.strip()

        # Basic validation: check if it starts with 'digraph' or 'graph'
        if dot_code.lower().startswith('digraph') or dot_code.lower().startswith('graph'):
             # Remove potential markdown backticks if LLM included them despite instructions
            if dot_code.startswith("```dot"):
                dot_code = dot_code[6:]
            if dot_code.endswith("```"):
                dot_code = dot_code[:-3]
            return dot_code.strip()
        elif "NO_FLOWCHART" in dot_code:
             return "NO_FLOWCHART"
        else:
             st.warning("LLM did not return valid DOT code or 'NO_FLOWCHART'. Trying to interpret...")
             # Attempt cleanup if possible, otherwise return potentially problematic code
             if "digraph" in dot_code or "graph" in dot_code:
                 # Try extracting the dot part if buried
                 start = dot_code.find("digraph")
                 if start == -1: start = dot_code.find("graph")
                 end = dot_code.rfind("}") + 1 if start != -1 else -1
                 if start != -1 and end != -1:
                     return dot_code[start:end].strip()
             return "INVALID_DOT" # Indicate failure more clearly

    except Exception as e:
        st.error(f"Error generating flowchart DOT code: {e}")
        return "ERROR_GENERATING_DOT"

def render_flowchart(dot_code):
    """Renders DOT code using Graphviz."""
    if not dot_code or dot_code in ["NO_FLOWCHART", "INVALID_DOT", "ERROR_GENERATING_DOT"]:
        if dot_code == "NO_FLOWCHART":
            st.info("The LLM determined that a flowchart representation is not suitable for this document's content.")
        elif dot_code == "INVALID_DOT":
            st.warning("The LLM output could not be reliably interpreted as DOT code. Cannot render flowchart.")
        elif dot_code == "ERROR_GENERATING_DOT":
             st.error("An error occurred during flowchart code generation.")
        return None # Return None if no valid graph to render

    try:
        # Use st.graphviz_chart to render directly in Streamlit
        graph = graphviz.Source(dot_code) # Create the Source object
        st.graphviz_chart(dot_code)
        # Store the *Source object* itself for download
        st.session_state.graph_object = graph
        return True
    except graphviz.backend.execute.CalledProcessError as e:
        st.error(f"Graphviz rendering error: {e}. Please ensure Graphviz is installed and in your system PATH.")
        st.code(dot_code, language='dot') # Show the code that failed
        return False # Indicate failure
    except Exception as e:
        st.error(f"An unexpected error occurred during flowchart rendering: {e}")
        st.code(dot_code, language='dot')
        return False # Indicate failure

def setup_qa_chain(_llm, _vector_store):
    """Sets up the RetrievalQA chain."""
    if _vector_store is None:
        return None
    # Define a prompt template for QA
    qa_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_template)

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        _llm,
        retriever=_vector_store.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=False # Set to True if you want to see source chunks
    )
    return qa_chain

# --- Download Functions ---
def download_text_button(text_content, filename, button_text="Download Text"):
    """Creates a download button for text content."""
    # Create a download button
    b64 = base64.b64encode(text_content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="text-decoration:none;">'\
           f'<button style="background-color:#4CAF50;color:white;padding:8px 15px;'\
           f'border:none;border-radius:5px;cursor:pointer;">{button_text}</button></a>'
    return href

def get_image_download_link(graph, filename="flowchart.png", format="png", button_text="Download Flowchart"):
    """Generate a download link for the rendered graph using graph.pipe()."""
    if not isinstance(graph, graphviz.Source):
         st.error("Invalid graph object provided for download.")
         return "<p>Error: Cannot generate download link for invalid graph object.</p>"
    try:
        # Use graph.pipe() to get the image bytes directly
        # Ensure the format requested is supported by graphviz (png, svg, pdf, etc.)
        img_bytes = graph.pipe(format=format)

        if not img_bytes:
            st.error("Graphviz returned empty output. Cannot create download link.")
            return "<p>Error: Graphviz rendering failed (empty output).</p>"

        # Base64 encode the image data
        b64 = base64.b64encode(img_bytes).decode()
        mime_type = f"image/{format}" # Adjust mime type based on format if needed
        href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" '\
              f'style="text-decoration:none;"><button style="background-color:#4CAF50;'\
              f'color:white;padding:8px 15px;border:none;border-radius:5px;cursor:pointer;">'\
              f'{button_text}</button></a>'
        return href
    except graphviz.backend.execute.CalledProcessError as e:
        # Handle potential errors if graphviz executable isn't found or has issues
        st.error(f"Graphviz execution error during download link generation: {e}")
        return f"<p>Error creating download link: Graphviz execution failed ({e}).</p>"
    except Exception as e:
        st.error(f"Unexpected error creating download link: {e}")
        return f"<p>Error creating download link: {e}</p>"

def download_chat_history(chat_history, filename="chat_history.json"):
    """Creates a download button for chat history in JSON format."""
    # Convert chat history to JSON
    json_str = json.dumps(chat_history, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}" style="text-decoration:none;">'\
           f'<button style="background-color:#4CAF50;color:white;padding:8px 15px;'\
           f'border:none;border-radius:5px;cursor:pointer;">Download Chat History</button></a>'
    return href

# --- Main App Logic ---

# Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="AI PDF Analyzer", layout="wide")
st.title("📄 AI-Powered PDF Analyzer & QA Bot")
st.markdown("Upload a PDF to get a summary, visualize its structure (if applicable), and ask questions about its content.")

# Initialize session state variables
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
    st.session_state.texts = None
    st.session_state.full_text = None
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.summary = None
    st.session_state.dot_code = None
    st.session_state.flowchart_rendered = False
    st.session_state.graph_object = None
    st.session_state.chat_history = []
    st.session_state.pdf_name = "document.pdf"  # Initialize with a default value

# --- Sidebar for PDF Upload and LLM selection ---
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    # Add LLM selection (optional, defaulting to llama3)
    llm_model_name = st.selectbox(
        "Select LLM Model (Groq)",
        # Add more models supported by Groq as needed
        ("llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"),
        index=0 # Default to llama3-8b
    )

    process_button = st.button("Process PDF", disabled=not uploaded_file)

# --- Initialize LLM and Embeddings ---
# Moved initialization outside the button click to avoid re-init on every interaction
# Use caching for Embeddings model loading
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(_model_name, _api_key):
     return ChatGroq(temperature=0.2, groq_api_key=_api_key, model_name=_model_name)

embeddings = get_embeddings_model()
llm = get_llm(llm_model_name, groq_api_key)

# --- Processing Logic ---
if process_button and uploaded_file:
    # Reset state if a new file is processed
    st.session_state.pdf_processed = False
    st.session_state.texts = None
    st.session_state.full_text = None
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.summary = None
    st.session_state.dot_code = None
    st.session_state.flowchart_rendered = False
    st.session_state.graph_object = None
    st.session_state.chat_history = []
    # Store the filename
    if uploaded_file and hasattr(uploaded_file, 'name'):
        st.session_state.pdf_name = uploaded_file.name
    else:
        st.session_state.pdf_name = "document.pdf"  # Fallback name

    # Load, split, and embed
    texts, full_text = load_and_split_pdf(uploaded_file)
    if texts and full_text and embeddings:
        st.session_state.texts = texts
        st.session_state.full_text = full_text
        st.session_state.vector_store = create_vector_store(texts, embeddings)

        if st.session_state.vector_store:
            # Setup QA chain
            st.session_state.qa_chain = setup_qa_chain(llm, st.session_state.vector_store)

            # Generate Summary and Flowchart in parallel (conceptually)
            st.session_state.summary = get_summary(llm, st.session_state.full_text)
            st.session_state.dot_code = get_flowchart_dot(llm, st.session_state.full_text)

            st.session_state.pdf_processed = True # Mark as processed
            st.success("PDF Processed Successfully!")
        else:
            st.error("Failed to create vector store. Cannot proceed with analysis.")
            st.session_state.pdf_processed = False
    else:
        st.error("Failed to load or process the PDF.")
        st.session_state.pdf_processed = False


# --- Display Results (only if PDF processed) ---
if st.session_state.pdf_processed:
    st.markdown("---")
    st.header("Analysis Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Summary")
        if st.session_state.summary:
            st.info(st.session_state.summary)
            
            # Create a timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_basename = os.path.splitext(st.session_state.pdf_name)[0]
            summary_filename = f"{pdf_basename}_summary_{timestamp}.txt"
            
            # Add download button
            st.markdown(download_text_button(st.session_state.summary, summary_filename, "Download Summary"), unsafe_allow_html=True)
        else:
            st.warning("Summary could not be generated.")

    with col2:
        st.subheader("📊 Flowchart / Structure")
        # Render flowchart if not already rendered
        if 'flowchart_rendered' not in st.session_state or not st.session_state.flowchart_rendered:
            st.session_state.flowchart_rendered = render_flowchart(st.session_state.dot_code) # Attempt rendering

        # Add download button for flowchart if it was rendered successfully
        if st.session_state.flowchart_rendered and st.session_state.graph_object:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_basename = os.path.splitext(st.session_state.pdf_name)[0]
            flowchart_filename = f"{pdf_basename}_flowchart_{timestamp}.png"
            
            # Make sure to render the download button
            st.markdown(get_image_download_link(st.session_state.graph_object, flowchart_filename), unsafe_allow_html=True)

        # Add a button to show and download raw DOT code if available
        if st.session_state.dot_code and st.session_state.dot_code not in ["NO_FLOWCHART", "ERROR_GENERATING_DOT"]:
            exp = st.expander("Show Flowchart DOT Code")
            with exp:
                st.code(st.session_state.dot_code, language='dot')
                
                # Download button for DOT code
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_basename = os.path.splitext(st.session_state.pdf_name)[0]
                dot_filename = f"{pdf_basename}_flowchart_{timestamp}.dot"
                st.markdown(download_text_button(st.session_state.dot_code, dot_filename, "Download DOT Code"), unsafe_allow_html=True)


    st.markdown("---")
    st.header("❓ Ask Questions about the PDF")

    if st.session_state.qa_chain:
        user_question = st.text_input("Enter your question:")
        ask_button = st.button("Ask")

        if ask_button and user_question:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke({"query": user_question})
                    answer = response.get("result", "Sorry, I couldn't find an answer.")
                    
                    # Add Q&A to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    st.info("Answer:")
                    st.write(answer)

                except Exception as e:
                    st.error(f"Error getting answer: {e}")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, qa in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{i+1}: {qa['question']}**")
                st.markdown(f"*A: {qa['answer']}*")
                st.markdown("---")
            
            # Add download button for chat history
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_basename = os.path.splitext(st.session_state.pdf_name)[0]
            chat_filename = f"{pdf_basename}_chat_history_{timestamp}.json"
            st.markdown(download_chat_history(st.session_state.chat_history, chat_filename), unsafe_allow_html=True)
    else:
        st.warning("QA system not initialized. Please process a PDF first.")

elif not uploaded_file:
    st.info("Please upload a PDF file using the sidebar to begin.")