from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


def generate_response(llm, question: str, docs: list):
  context = "\n\n".join(doc.page_content for doc in docs)
  
  system_prompt_template = SystemMessagePromptTemplate.from_template("""
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Only return the helpful answer below and nothing else.
  """)
  user_prompt_template = HumanMessagePromptTemplate.from_template("""
    Context: {context}
    Question: {question}
  """)
  
  prompt_template = ChatPromptTemplate.from_messages([system_prompt_template, user_prompt_template])
  chain = LLMChain(llm, prompt_template)

  response = chain.invoke({"context": context, "question": question})
  return response


def generate_subquestions(llm, question: str):
  system_prompt_template = SystemMessagePromptTemplate.from_template("""
    You are a helpful assistant that generates multiple search queries based on a single input query.
    """)
  user_prompt_template = HumanMessagePromptTemplate.from_template("""
    Generate multiple search queries related to: {question}
    OUTPUT (4 queries):
  """)
  
  prompt_template = ChatPromptTemplate.from_messages([system_prompt_template, user_prompt_template])
  chain = LLMChain(llm, prompt_template)

  response = chain.invoke({"question": question})
  
  generated_queries = response.choices[0]["message"]["content"].strip().split("\n")
  return generated_queries  