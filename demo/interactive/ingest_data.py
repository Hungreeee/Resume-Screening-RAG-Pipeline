# %%
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
FAISS_PATH = os.getenv("FAISS_PATH")

# %%
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

df = pd.read_csv(DATA_PATH)
loader = DataFrameLoader(df, page_content_column="Resume")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1024,
    chunk_overlap = 500
)

documents = loader.load()
document_chunks = text_splitter.split_documents(documents)

# %%
from time import time

start = time()
vectorstore_db = FAISS.from_documents(document_chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE)
vectorstore_db.save_local(FAISS_PATH)
end = time()
print(f"{end - start} seconds")
