import numpy
import pandas
import pymupdf
import chromadb
import streamlit
import langgraph
import llama_index


def main():
    print("================================")
    print("      SPACE RAG ENVIRONMENT")
    print("================================")

    print("NumPy       :", numpy.__version__)
    print("Pandas      :", pandas.__version__)
    print("PyMuPDF     :", pymupdf.__version__)
    print("ChromaDB    :", chromadb.__version__)
    print("Streamlit   :", streamlit.__version__)

    print("\nEnvironment is working!")


if __name__ == "__main__":
    main()