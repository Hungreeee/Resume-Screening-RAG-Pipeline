# Resume Screening Chatbot using RAG 

## Introduction

The project is a proof of concept (POC) of an LLM-based assistant to help hiring managers screen resumes more efficiently. The key design integrates Retrieval Augmented Generation (RAG) Fusion to effectively retrieve the top matching resumes from a large pool of applicants for a job description. This data is then augmented into the LLM for downstream tasks like analysis, summarization, or decision-making. 

## Demo

The app will be hosted soon.

![Screenshot_118](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/assets/46376260/11c78009-af1e-4cab-9617-2a16e618e7d3)

![Screenshot_119](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/assets/46376260/991aee26-af7c-440f-b050-f5789aff3d84)

## Program Structure

The system uses RAG Fusion, an advanced RAG framework that combines generative agents and similarity-based retrieval methods to enhance answer quality. This is particularly useful in resume screening, where queries often contain complex and multifaceted job requirements.

![system-structure](https://github.com/Hungreeee/Resume-Screening-LLM-RAG/assets/46376260/b108cbda-81fa-495c-b2a6-c3a279310bf6)

The process begins by processing resumes into a vector storage. Upon receiving the input job descriptions query, the LLM agent is prompted to generate sub-queries. The vector storage then performs a retrieval process for each given query to return the top-K most similar documents. The document list for each sub-query is then combined and re-ranked into a new list, representing the most similar documents to the original job description. The LLM then utilizes the retrieved applicants' information as context to form accurate, relevant, and informative responses to assist hiring managers in matching resumes with job descriptions.

Tech stacks: 
- `langchain`, `openai`: Chatbot construction.
- `faiss`: Vector indexing and similarity retrieval.
- `streamlit`: Chatbot interface development.

## Installation and Setup

To install the project:
```
git clone https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline.git
pip install requirements.txt
```

To run the program locally:
```
streamlit run demo/interface.py
```

