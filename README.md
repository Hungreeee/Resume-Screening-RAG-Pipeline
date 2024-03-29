# Introduction

The overall goal is to propose a proof of concept (POC) of an LLM-based assistant to aid hiring managers in screening resumes more efficiently through question-answering. The key design is to integrate Retrieval Augmented Generation (RAG) to effectively retrieve the top matching resumes from a large pool of applicants and augment them to the LLM's knowledge base, which enables the generation of high-level, accurate, and relevant responses about the applicants to the recruiter's queries.

# Proprosed POC

The system employs RAG Fusion, an advanced RAG framework that combines generative agents and similarity-based retrieval methods to enhance answer quality. This is particularly useful in resume screening, where queries often contain complex and multifaceted job requirements.

![system-structure](https://github.com/Hungreeee/Resume-Screening-LLM-RAG/assets/46376260/b108cbda-81fa-495c-b2a6-c3a279310bf6)

The process begins by processing resumes into a vector storage. Upon receiving the input job descriptions query, the LLM agent is prompted to generate sub-queries. The vector storage then performs a retrieval process for each given query to return the top-K most similar documents. The document list for each sub-query is combined and re-ranked into a ranked document list. This document set is then augmented into an LLM agent for the generative process. The LLM then utilizes the retrieved applicants' information as context to form accurate, relevant, and informative responses to assist hiring managers in matching resumes with job descriptions.

Tech stacks: `langchain`, `FAISS`, `openai`.

# Evaluation

