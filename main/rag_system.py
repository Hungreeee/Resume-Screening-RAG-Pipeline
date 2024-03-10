import sys, httpx, os
sys.dont_write_bytecode = True

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from llm_utils import generate_response, generate_subquestions


DATA_PATH = "./data/alt-cleaned_resume.csv"
FAISS_PATH = "./alt-vectorstore"
RAG_K_THRESHOLD = 2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-35-turbo"
OPENAI_ENDPOINT = "https://aalto-openai-apigw.azure-api.net"
OPENAI_KEY = ""


def update_base_url(request: httpx.Request):
  if request.url.path == "/chat/completions":
    request.url = request.url.copy_with(path="/v1/chat")


def reciprocal_rank_fusion(document_rank_list: list[dict], k=50):
  fused_scores = {}
  for doc_list in document_rank_list:
    for rank, (doc, score) in enumerate(doc_list.items()):
      if doc not in fused_scores:
        fused_scores[doc] = 0
      fused_scores[doc] += 1 / (rank + k)
  reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
  return reranked_results


def retrieve_docs_id(question: str, k=3):
  docs_score = vectorstore_db.similarity_search_with_score(question, k=k)
  print(docs_score)
  docs_score = {str(doc.metadata["ID"]): score for doc, score in docs_score}
  return docs_score


def retrieve_id_and_rerank(subquestion_list: list):
  document_rank_list = []
  for subquestion in subquestion_list:
    document_rank_list.append(retrieve_docs_id(subquestion, RAG_K_THRESHOLD))
  reranked_documents = reciprocal_rank_fusion(document_rank_list)
  return reranked_documents


def retrieve_documents_with_id(documents: pd.DataFrame, doc_id_with_score: dict):
  id_list = list(doc_id_with_score.keys())
  retrieved_documents = list(documents[documents["ID"].isin(id_list)]['Resume'])
  return retrieved_documents 


if __name__ == "__main__":
  documents = pd.read_csv(DATA_PATH)
  documents["ID"] = documents["ID"].astype('str')

  embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
  )

  llm = ChatOpenAI(
    default_headers={"Ocp-Apim-Subscription-Key": OPENAI_KEY},
    base_url="https://aalto-openai-apigw.azure-api.net",
    api_key=False,
    http_client=httpx.Client(
      event_hooks={
        "request": [update_base_url],
    }),
  )

  vectorstore_db = FAISS.load_local(FAISS_PATH, embedding_model, distance_strategy=DistanceStrategy.COSINE)
  id_list = retrieve_id_and_rerank(["Find front-end developer with experience in database management."])
  doc_list = retrieve_documents_with_id(documents, id_list)

  while True:
    question = str(input("Question: "))
    subquestion_list = generate_subquestions(llm, question)
    document_list = retrieve_and_rerank(subquestion_list)
    response = generate_response(llm, question, document_list)
    print(response)