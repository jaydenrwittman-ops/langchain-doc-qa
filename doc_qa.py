"""
========================================================
  Document Question-Answering System
  Built with LangChain + OpenAI + FAISS
========================================================

HOW TO SET UP:
--------------
1. Install dependencies:
      pip install langchain langchain-openai langchain-community \
                  langchain-text-splitters faiss-cpu pypdf python-dotenv openai

2. Create a file named `.env` in the same folder as this script:
      OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

3. Run the script:
      python doc_qa.py --file your_document.pdf
      python doc_qa.py --file your_document.txt

HOW IT WORKS (high-level):
---------------------------
  Load document → Split into chunks → Embed chunks → Store in FAISS
  → Accept your question → Retrieve relevant chunks → GPT answers

INTERVIEW TALKING POINTS:
--------------------------
  • RAG (Retrieval-Augmented Generation): grounds LLM answers in your doc
  • Chunking: prevents hitting token limits; overlapping chunks preserve context
  • Vector store: semantic similarity search (not keyword matching)
  • LCEL pipeline: modern LangChain Expression Language chain composition
"""

import os
import sys
import argparse
from pathlib import Path

# ── Environment ───────────────────────────────────────────────────────────────

from dotenv import load_dotenv

# Load OPENAI_API_KEY from .env file in the current directory
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print(
        "\n❌  OPENAI_API_KEY not found.\n"
        "   Create a .env file in this directory with:\n"
        "       OPENAI_API_KEY=sk-your-key-here\n"
    )
    sys.exit(1)

# ── LangChain imports (modern, post-0.2 style) ───────────────────────────────

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load the document
# ══════════════════════════════════════════════════════════════════════════════

def load_document(file_path: str) -> list:
    """
    Load a PDF or TXT file and return a list of LangChain Document objects.
    Each Document has .page_content (the text) and .metadata (source info).
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    print(f"\n📄  Loading '{path.name}' ...")

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Please provide a .pdf or .txt file."
        )

    documents = loader.load()
    print(f"   ✅  Loaded {len(documents)} page(s) / section(s).")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Split into chunks
# ══════════════════════════════════════════════════════════════════════════════

def split_documents(documents: list) -> list:
    """
    Split documents into smaller, overlapping chunks.

    Why chunk?
    • Embedding models and LLMs have token limits — large pages won't fit.
    • Smaller chunks = more precise retrieval.

    Why overlap?
    • chunk_overlap=200 means the last 200 characters of chunk N reappear at
      the start of chunk N+1, so sentences at boundaries aren't lost.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"\n✂️   Split into {len(chunks)} chunk(s) "
          f"(chunk_size=1000, overlap=200).")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Create embeddings and build a FAISS vector store
# ══════════════════════════════════════════════════════════════════════════════

def build_vector_store(chunks: list) -> FAISS:
    """
    Convert each chunk into a vector embedding and store in FAISS.

    • OpenAIEmbeddings uses text-embedding-ada-002 (1536-dimensional vectors).
    • FAISS enables fast nearest-neighbor search across all chunk vectors.
    • When you ask a question, it's also embedded and compared to these vectors
      to find the most semantically relevant chunks.
    """
    print("\n🔢  Generating embeddings and building FAISS index ...")
    print("   (Calls the OpenAI API — may take a few seconds.)")

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-ada-002",
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    print("   ✅  Vector store ready.")
    return vector_store


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Build the RAG chain (modern LCEL style)
# ══════════════════════════════════════════════════════════════════════════════

def build_qa_chain(vector_store: FAISS):
    """
    Build a Retrieval-Augmented Generation (RAG) chain using modern
    LangChain Expression Language (LCEL).

    Flow for each question:
      1. Retriever: embed the question → find top-4 similar chunks in FAISS
      2. Prompt: insert those chunks as context into a prompt template
      3. LLM: GPT-3.5-turbo reads the context and generates an answer
      4. Parser: extracts the plain string from the response object
    """
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0,       # 0 = deterministic / factual answers
        max_tokens=512,
    )

    # Retrieve the 4 most relevant chunks for each question
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # Prompt template — {context} gets filled with retrieved chunks,
    # {question} gets filled with the user's question
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context from a document to 
answer the question. If the answer isn't in the context, say so honestly.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        """Join retrieved chunk texts into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain: retrieve → format → prompt → LLM → parse output
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Interactive Q&A loop
# ══════════════════════════════════════════════════════════════════════════════

def run_qa_loop(chain) -> None:
    """
    Accept questions from the user in a loop and print GPT's answers.
    Type 'quit' or 'exit' to stop.
    """
    print("\n" + "═" * 60)
    print("  💬  Ask questions about your document.")
    print("  Type  'quit'  or  'exit'  to stop.")
    print("═" * 60)

    while True:
        try:
            question = input("\n❓  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  Goodbye!")
            break

        if not question:
            print("   ⚠️  Please enter a question.")
            continue

        if question.lower() in {"quit", "exit"}:
            print("\n👋  Goodbye!")
            break

        try:
            answer = chain.invoke(question)
            print("\n🤖  Answer:")
            print("   " + answer.strip())

        except Exception as api_err:
            print(f"\n❌  Error: {api_err}")
            print("   Check your API key and that you have OpenAI credits.")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about a PDF or TXT file using GPT-3.5-turbo."
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the .pdf or .txt document you want to query.",
    )
    args = parser.parse_args()

    try:
        documents    = load_document(args.file)       # Step 1
        chunks       = split_documents(documents)      # Step 2
        vector_store = build_vector_store(chunks)      # Step 3
        chain        = build_qa_chain(vector_store)    # Step 4
        run_qa_loop(chain)                             # Step 5

    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌  {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌  Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()