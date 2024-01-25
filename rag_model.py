from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate


FAISS_PATH = "./vectorstore/faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = ""


def reciprocal_rank_fusion(search_results_dict, k=50):
    fused_scores = {}
    # print("Initial individual search result ranks:")
    # for query, doc_scores in search_results_dict.items():
    #     print(f"For query '{query}': {doc_scores}")
        
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
            # print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    print("Final reranked results:", reranked_results)
    return reranked_results


def retrieve_docs(retriever, question, k=50):
  docs_score = retriever.similarity_search_with_score(question, k=k)
  docs_score = {"Applicant ID: " + str(doc.metadata["ID"]) + "\n" + doc.page_content: score for doc, score in docs_score}
  docs_score_sorted = {doc: score for doc, score in sorted(docs_score.items(), key=lambda x: x[1], reverse=True)}
  return docs_score_sorted


if __name__ == "__main__":
  embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
  )
  vectorstore_db = FAISS.load_local(FAISS_PATH, embedding_model)
  docs = retrieve_docs(vectorstore_db, "Data Scientist", 2)
  print(docs)