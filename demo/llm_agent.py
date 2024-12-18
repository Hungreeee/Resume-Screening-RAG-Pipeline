import sys, httpx, os
sys.dont_write_bytecode = True

from dotenv import load_dotenv

from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
RAG_K_THRESHOLD = 5
LLM_MODEL = "gpt-35-turbo"
CUSTOMED_ENDPOINT = "https://aalto-openai-apigw.azure-api.net"


class ChatBot():
  def __init__(self, api_key: str, model: str):
    self.llm = ChatOpenAI(
      model=model, 
      api_key=api_key, 
      temperature=0.1
    )

  def generate_subquestions(self, question: str):
    system_message = SystemMessage(content="""
      You are an expert in talent acquisition. Separate this job description into 3-4 more focused aspects for efficient resume retrieval. 
      Make sure every single relevant aspect of the query is covered in at least one query. You may choose to remove irrelevant information that doesn't contribute to finding resumes such as the expected salary of the job, the ID of the job, the duration of the contract, etc.
      Only use the information provided in the initial query. Do not make up any requirements of your own.
      Put each result in one line, separated by a linebreak.
      """)
    
    user_message = HumanMessage(content=f"""
      Generate 3 to 4 sub-queries based on this initial job description: 
      {question}
    """)

    oneshot_example = HumanMessage(content="""
      Generate 3 to 4 sub-queries based on this initial job description:

      Wordpress Developer
      We are looking to hire a skilled WordPress Developer to design and implement attractive and functional websites and Portals for our Business and Clients. You will be responsible for both back-end and front-end development including the implementation of WordPress themes and plugins as well as site integration and security updates.
      To ensure success as a WordPress Developer, you should have in-depth knowledge of front-end programming languages, a good eye for aesthetics, and strong content management skills. Ultimately, a top-class WordPress Developer can create attractive, user-friendly websites that perfectly meet the design and functionality specifications of the client.
      WordPress Developer Responsibilities:
      Meeting with clients to discuss website design and function.
      Designing and building the website front-end.
      Creating the website architecture.
      Designing and managing the website back-end including database and server integration.
      Generating WordPress themes and plugins.
      Conducting website performance tests.
      Troubleshooting content issues.
      Conducting WordPress training with the client.
      Monitoring the performance of the live website.
      WordPress Developer Requirements:
      Bachelors degree in Computer Science or a similar field.
      Proven work experience as a WordPress Developer.
      Knowledge of front-end technologies including CSS3, JavaScript, HTML5, and jQuery.
      Knowledge of code versioning tools including Git, Mercurial, and SVN.
      Experience working with debugging tools such as Chrome Inspector and Firebug.
      Good understanding of website architecture and aesthetics.
      Ability to project manage.
      Good communication skills.
      Contract length: 12 months
      Expected Start Date: 9/11/2020
      Job Types: Full-time, Contract
      Salary: 12,004.00 - 38,614.00 per month
      Schedule:
      Flexible shift
      Experience:
      Wordpress: 3 years (Required)
      web designing: 2 years (Required)
      total work: 3 years (Required)
      Education:
      Bachelor's (Preferred)
      Work Remotely:
      Yes
    """)

    oneshot_response = AIMessage(content="""
      1. WordPress Developer Skills:
        - WordPress, front-end technologies (CSS3, JavaScript, HTML5, jQuery), debugging tools (Chrome Inspector, Firebug), code versioning tools (Git, Mercurial, SVN).
        - Required experience: 3 years in WordPress, 2 years in web designing.
    
      2. WordPress Developer Responsibilities:
        - Meeting with clients for website design discussions.
        - Designing website front-end and architecture.
        - Managing website back-end including database and server integration.
        - Generating WordPress themes and plugins.
        - Conducting website performance tests and troubleshooting content issues.
        - Conducting WordPress training with clients and monitoring live website performance.

      3. WordPress Developer Other Requirements:
        - Education requirement: Bachelor's degree in Computer Science or similar field.
        - Proven work experience as a WordPress Developer.
        - Good understanding of website architecture and aesthetics.
        - Ability to project manage and strong communication skills.

      4. Skills and Qualifications:
        - Degree in Computer Science or related field.
        - Proven experience in WordPress development.
        - Proficiency in front-end technologies and debugging tools.
        - Familiarity with code versioning tools.
        - Strong communication and project management abilities.
    """)

    response = self.llm.invoke([system_message, oneshot_example, oneshot_response, user_message])
    result = response.content.split("\n\n")
    return result

  def generate_message_stream(self, question: str, docs: list, history: list, prompt_cls: str):
    context = "\n\n".join(doc for doc in docs)
    
    if prompt_cls == "retrieve_applicant_jd":
      system_message = SystemMessage(content="""
        You are an expert in talent acquisition that helps determine the best candidate among multiple suitable resumes.
        Use the following pieces of context to determine the best resume given a job description. 
        You should provide some detailed explanations for the best resume choice.
        Because there can be applicants with similar names, use the applicant ID to refer to resumes in your response. 
        If you don't know the answer, just say that you don't know, do not try to make up an answer.
      """)

      user_message = HumanMessage(content=f"""
        Chat history: {history}
        Context: {context}
        Question: {question}
      """)

    else:
      system_message = SystemMessage(content="""
        You are an expert in talent acquisition that helps analyze resumes to assist resume screening.
        You may use the following pieces of context and chat history to answer your question. 
        Do not mention in your response that you are provided with a chat history.
        If you don't know the answer, just say that you don't know, do not try to make up an answer.
      """)

      user_message = HumanMessage(content=f"""
        Chat history: {history}
        Question: {question}
        Context: {context}
      """)

    stream = self.llm.stream([system_message, user_message])
    return stream