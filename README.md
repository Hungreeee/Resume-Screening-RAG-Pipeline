# Resume Screening RAG Pipeline

## Introduction

The project is a proof of concept for an LLM assistant aiming to assist hiring managers in resume screening. The core design of the assistant involves the use of Retrieval Augmented Generation (RAG) Fusion:

1. Retrieval: When a job description is provided, the retriever searches for similar resumes. This is done to narrow the pool of applicants to the most relevant profiles.
2. Generation: The retrieved resumes are then used to augment the LLM generator so it is conditioned on the data of the top applicants given the job description. The generator can then be used for further downstream tasks like analysis, summarization, or decision-making.

#### Why resume screening?

Despite the increasingly large volume of applicants each year, there are limited tools that can assist the screening process effectively and reliably. Existing methods often revolve around keyword-based approaches, which cannot accurately handle the complexity of natural language in human-written documents. Because of this, there is a clear opportunity to integrate LLM-based methods into this domain, which the project aims to address. 

#### Why RAG/RAG Fusion? 

RAG-like frameworks are great tools to enhance the reliability of chatbots. Overall, RAG aims to provide an external knowledge base for LLM agents, allowing them to receive additional context relevant to user queries. This increases the relevance and accuracy of the generated answers, which is especially important in data-intensive environments such as the recruitment domain.

On the other hand, RAG Fusion is effective in addressing complex and ambiguous human-written prompts. While the LLM generator can handle this problem effectively, the retriever may struggle to find relevant documents when presented with multifaceted queries. Therefore, this technique can be used to improve resume retrieval quality when the system receives complex job descriptions (which are quite common in hiring).

> [!NOTE]
> For more info, please refer to the paper: [Google Drive](https://drive.google.com/drive/folders/19pL-MNfPUVsxePHd8FDvddnoC3S_zNFN?usp=drive_link)

## Demo

The demo interface of the chatbot can be found here: [Streamlit](https://resume-screening-rag-gpt.streamlit.app)

Default synthetic resume data set used in the demo: [GitHub](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/main-data/synthetic-resumes.csv)

Source job description dataset: [Kaggle](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description)

> [!WARNING]
> The file uploader is still quite unstable in Streamlit deployment. I do not recommend using it.

**Starting screen:**

![Screenshot_121](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/assets/46376260/b585d5da-0e19-4f70-8735-19f18b83080c)

**The system's example response when receiving a job description:**

![Screenshot_119](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/assets/46376260/991aee26-af7c-440f-b050-f5789aff3d84)

## System Description

### 1. RAG Pipeline 

![system-structure](https://github.com/Hungreeee/Resume-Screening-LLM-RAG/assets/46376260/b108cbda-81fa-495c-b2a6-c3a279310bf6)

The process begins by processing resumes into a vector storage. Upon receiving the input job descriptions query, the LLM agent is prompted to generate sub-queries. The vector storage then performs a retrieval process for each given query to return the top-K most similar documents. The document list for each sub-query is then combined and re-ranked into a new list, representing the most similar documents to the original job description. The LLM then utilizes the retrieved applicants' information as context to form accurate, relevant, and informative responses to assist hiring managers in matching resumes with job descriptions.

### 2. Chatbot

In addition to the original RAG pipeline, the deployed chatbot utilizes certain techniques to be more suitable for real-world use cases:

- Chat history access: The LLM is fed with the entire conversation and the (latest) retrieved documents for every message, allowing it to perform follow-up tasks. 
- Query classification: The LLM checks whether the query contains a job description to toggle RAG on/off. The system only performs RAG when a suitable query is fed. Otherwise, it is directly fed with the chat history to address follow-up prompts.
- Small-to-Big retrieval: The retrieval process is performed using text chunks for efficiency. The retrieved chunks are then traced back to their original full-text documents to augment the LLM generator, allowing the generator to receive the complete context of the resumes. 

**Tech stacks:** 
- `langchain`, `openai`, `huggingface`: RAG pipeline and chatbot construction.
- `faiss`: Vector indexing and similarity retrieval.
- `streamlit`: User interface development.

## Installation and Setup

To set up the project locally:
```
# Clone the project
git clone https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline.git

# Install dependencies
pip install requirements.txt
```

To run the Streamlit demo locally:
```
streamlit run demo/interface.py
```

## Contributions

The design of the deployed chatbot is relatively simple because it only serves to show the bigger picture of the potential of RAG-like systems in the recruitment domain. As such, the system is still very much a work in progress and any suggestion/contribution is highly appreciated! Please share them at [Issue](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/issues). 

## Acknowledgement

- Inspired by [RAG Fusion](https://github.com/Raudaschl/rag-fusion) 
