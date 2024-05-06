import sys
sys.dont_write_bytecode = True


DATA_PATH = "./data/main-data/synthetic-resumes.csv"
FAISS_PATH = "./vectorstore"
RAG_K_THRESHOLD = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-35-turbo"
OPENAI_ENDPOINT = "https://aalto-openai-apigw.azure-api.net"


class RAGPipeline():
  def __init__(self, vectorstore_db, df):
    self.vectorstore = vectorstore_db
    self.df = df

  def __reciprocal_rank_fusion__(self, document_rank_list: list[dict], k=50):
    fused_scores = {}
    for doc_list in document_rank_list:
      for rank, (doc, _) in enumerate(doc_list.items()):
        if doc not in fused_scores:
          fused_scores[doc] = 0
        fused_scores[doc] += 1 / (rank + k)
    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results


  def __retrieve_docs_id__(self, question: str, k=50):
    docs_score = self.vectorstore.similarity_search_with_score(question, k=k)
    docs_score = {str(doc.metadata["ID"]): score for doc, score in docs_score}
    return docs_score


  def retrieve_id_and_rerank(self, subquestion_list: list):
    document_rank_list = []
    for subquestion in subquestion_list:
      document_rank_list.append(self.__retrieve_docs_id__(subquestion, RAG_K_THRESHOLD))
    reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
    return reranked_documents


  def retrieve_documents_with_id(self, doc_id_with_score: dict, threshold=5):
    id_resume_dict = dict(zip(self.df["ID"].astype(str), self.df["Resume"]))
    retrieved_ids = list(sorted(doc_id_with_score, key=doc_id_with_score.get, reverse=True))[:threshold]
    retrieved_documents = [id_resume_dict[id] for id in retrieved_ids]
    for i in range(len(retrieved_documents)):
      retrieved_documents[i] = "Applicant ID " + retrieved_ids[i] + "\n" + retrieved_documents[i]
    return retrieved_documents 