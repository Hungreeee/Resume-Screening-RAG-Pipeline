import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


DATA_PATH = "./data/resume.csv"
FAISS_PATH = "./vectorstore/faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def initiate_vectorstore():
  df = pd.read_csv(DATA_PATH)
  loader = DataFrameLoader(df, page_content_column="Resume_str")
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 40,
  )
  embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
  )

  documents = loader.load()
  document_chunks = text_splitter.split_documents(documents)
  vectorstore_db = FAISS.from_documents(document_chunks, embedding_model)

  vectorstore_db.save_local(FAISS_PATH)


if __name__ == "__main__":
  initiate_vectorstore()