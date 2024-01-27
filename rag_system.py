from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from llm_utils import generate_response, generate_subquestions


FAISS_PATH = "./vectorstore"
RAG_K_THRESHOLD = 30
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"
OPENAI_KEY = ""


def reciprocal_rank_fusion(document_rank_list: list[dict], k=50):
  fused_scores = {}
  for doc_list in document_rank_list:
    for rank, (doc, score) in enumerate(doc_list.items()):
      if doc not in fused_scores:
        fused_scores[doc] = 0
      fused_scores[doc] += 1 / (rank + k)

  reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
  return reranked_results


def retrieve_docs(question: str, k=3):
  docs_score = vectorstore_db.similarity_search_with_score(question, k=k)
  docs_score = {"Applicant ID: " + str(doc.metadata["ID"]) + "\n" + doc.page_content: score for doc, score in docs_score}
  return docs_score


def retrieve_and_rerank(subquestion_list: list):
  document_rank_list = []
  for subquestion in subquestion_list:
    document_rank_list.append(retrieve_docs(subquestion, RAG_K_THRESHOLD))
  
  reranked_documents = reciprocal_rank_fusion(document_rank_list)
  return reranked_documents


if __name__ == "__main__":
  embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
  )
  llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1, openai_api_key=OPENAI_KEY)
  vectorstore_db = FAISS.load_local(FAISS_PATH, embedding_model)

  while True:
    question = str(input("Question: "))
    subquestion_list = generate_subquestions(llm, question)
    document_list = retrieve_and_rerank(subquestion_list)
    response = generate_response(llm, question, document_list)
    print(response)