import sys
sys.dont_write_bytecode = True

from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


def generate_response(llm, question: str, docs: list):
  context = "\n\n".join(doc for doc in docs)
  
  system_prompt_template = SystemMessagePromptTemplate.from_template("""
    You are an expert in talent acquisition that helps to determine the best candidates given their resumes as context.
    Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, do not try to make up an answer.
    Only return the helpful answer below and nothing else.
  """)
  user_prompt_template = HumanMessagePromptTemplate.from_template("""
    Context: {context}
    Question: {question}
    Answer:
  """)
  
  prompt_template = ChatPromptTemplate.from_messages([system_prompt_template, user_prompt_template])
  chain = LLMChain(llm=llm, prompt=prompt_template)

  response = chain.invoke({"context": context, "question": question})
  return response


def generate_subquestions(llm, question: str):
  system_prompt_template = SystemMessagePromptTemplate.from_template("""
    You are an expert in talent acquisition that generates sub-queries covering smaller but more focused aspects of the initial complex input job query.
    Only use the information provided in the initial query.                                                            
    """)
  user_prompt_template = HumanMessagePromptTemplate.from_template("""
    Generate 3 to 4 sub-queries based on this initial query and put all of the sub-queries in one line, separated by a semicolon: {question}
    Answer:
  """)
  
  prompt_template = ChatPromptTemplate.from_messages([system_prompt_template, user_prompt_template])
  chain = LLMChain(llm, prompt_template)

  response = chain.invoke({"question": question})
  
  generated_queries = response["content"].strip().split(";")
  return generated_queries  