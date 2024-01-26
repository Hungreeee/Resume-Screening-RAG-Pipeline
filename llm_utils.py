from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import openai


FAISS_PATH = "./vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = ""


def construct_prompt(question: str, docs: list):
  aggregated_context = "\n\n".join(doc.page_content for doc in docs)
  
  system_prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Only return the helpful answer below and nothing else.
    """
  user_prompt_template = f"""
    Context: {aggregated_context}
    Question: {question}
    """
  
  # (make call here)

  return (system_prompt_template, user_prompt_template)


def generate_subquestions(question: str):
  system_prompt_template = """
    You are a helpful assistant that generates multiple search queries based on a single input query.
    """
  user_prompt_template = f"""
    Generate multiple search queries related to: {question}
    OUTPUT (4 queries):
    """
  
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": system_prompt_template},
      {"role": "user", "content": user_prompt_template},
    ]
  )
  
  generated_queries = response.choices[0]["message"]["content"].strip().split("\n")
  return generated_queries  