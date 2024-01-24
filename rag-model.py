from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate


FAISS_PATH = "./vectorstore/faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = ""


def construct_prompt(question: str, context):
  aggregated_context = "\n\n".join(doc.page_content for doc in context)

  prompt = f"""
  Use the following pieces of information to answer the user's question.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.

  Context: {aggregated_context}
  Question: {question}

  Only return the helpful answer below and nothing else.
  Helpful answer:
  """

  return prompt


def model(query: str):
  embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
  )
  vectorstore_db = FAISS.load_local(FAISS_PATH, embedding_model)

  return


if __name__ == "__main__":
  model()