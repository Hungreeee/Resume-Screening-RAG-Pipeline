import langchain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate


FAISS_PATH = "./vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = ""


def reciprocal_rank_fusion(document_rank_list: list[list], k=50):
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


def re():
  
  return


if __name__ == "__main__":
  embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
  )
  vectorstore_db = FAISS.load_local(FAISS_PATH, embedding_model)
  docs = retrieve_docs("Data Science") 
  
  print(reciprocal_rank_fusion([docs]))