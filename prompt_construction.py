from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate


FAISS_PATH = "./vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = ""


def construct_prompt(question, docs):
  aggregated_context = "\n\n".join(doc.page_content for doc in docs)

  prompt_template = """
  Use the following pieces of information to answer the user's question.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.

  Context: {context}
  Question: {question}

  Only return the helpful answer below and nothing else.
  Helpful answer:
  """
  prompt = PromptTemplate(prompt_template, input_types=["context", "question"])

  return prompt.format(aggregated_context, question)


def generate_subquestion(question):
  
  return