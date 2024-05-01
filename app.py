#imports
import streamlit as st
from streamlit_mic_recorder import mic_recorder,speech_to_text
from google.cloud import storage
from google.cloud import speech
import webbrowser
import json
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
import requests
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
from trubrics.integrations.streamlit import FeedbackCollector
from newsapi import NewsApiClient

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =  "final_googleapp.json"
# Setting up environment for Google OAuth
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Definining your Google client ID, client secret,redirect URI, gemini api,newsapi
CLIENT_ID = st.secrets['CLIENT_ID']
CLIENT_SECRET = st.secrets['CLIENT_SECRET']
REDIRECT_URI = "http://localhost:8501/"
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
news_api_key = st.secrets['news_api_key']
newsapi = NewsApiClient(api_key=news_api_key)



llm = genai.GenerativeModel("gemini-1.0-pro-latest")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",task_type="retrieval_document")

collector = FeedbackCollector(
    email="prerna.chheda9811@gmail.com",
    password="Peru@9811",
    project="default"
)

llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest",temperature=0.5)
loader = WebBaseLoader(["https://cleartax.in/s/how-to-efile-itr","https://eportal.incometax.gov.in/iec/foservices/#/login","https://kpmg.com/in/en/home/services/tax/india-interim-budget-2024.html","https://financialservices.gov.in/beta/en","https://mgcub.ac.in/pdf/material/202004101433178abb4fd6e5.pdf","https://cleartax.in/s/income-tax-slabs",
                        "https://www.livemint.com/economy/budget-2024-25-key-highlights-live-updates-interim-budget-agriculture-infra-fiscal-deficit-nirmala-sitharaman-11706695416199.html"])
docs = loader.load()

# Splitting the text and document into chunks and storing it in vectorized db
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# Creating the vectorized db
db = DocArrayInMemorySearch.from_documents(
   docs, embeddings
)
trusted_websites =["https://eportal.incometax.gov.in/iec/foservices/#/login","https://cleartax.in/s/how-to-efile-itr","https://kpmg.com/in/en/home/services/tax/india-interim-budget-2024.html","https://financialservices.gov.in/beta/en","https://mgcub.ac.in/pdf/material/202004101433178abb4fd6e5.pdf","https://cleartax.in/s/income-tax-slabs",
                   "https://www.livemint.com/economy/budget-2024-25-key-highlights-live-updates-interim-budget-agriculture-infra-fiscal-deficit-nirmala-sitharaman-11706695416199.html"]
template = """You are a financial expert of India equipped with a vast knowledge of financial regulations, practices, and procedures, as well as access to specific details.
 When a user asks {question}, first look for the answer in {text}. 
If the answer is not present there, use your comprehensive understanding of Indian financial matters to provide an informed response. Ensure accuracy and reliability, especially when citing official websites, you can refer to websites like {trusted_websites} as an example and append your own based on India. 
Give answer using financial jargons and answer in simplified manner if asked.
Understand whenever user types financial slangs like LPA,CPS,CAPEX,etc.
If the person asks generic finance questions apart from {text} give answer based on India.
Frame your answers like you are casually chatting with them.
Apologize if the user is frustrated by your answer.
{chat_history}"""


prompt = PromptTemplate(template=template, input_variables=["chat_history", "text", "question"])
# Initializing  memory to story history of chat and giving context to followup questions
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
llm_chain = LLMChain(llm=llm, prompt=prompt,memory=memory)
# chain = load_qa_chain(llm_chain, chain_type="stuff", memory=memory)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain,memory=memory, document_variable_name="text")


def get_chatbot_response(user_input, chain, db):
    docs = db.similarity_search(user_input)
    chain_input = {
        "input_documents": docs,
        "context": "You are a financial advisor based in India using {input_documents}.",
        "trusted_websites": trusted_websites,
        # "context": """You are a financial advisor based in India answer various questions of users on basis of {CHAIN} """,
        "question": user_input
    }
    result = chain(chain_input, return_only_outputs=True)
    return result


def handle_chat_interaction(stuff_chain, db, collector):
    st.title("💰Wealth Whisperer💰")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    if 'transcribed_text' not in st.session_state:
        st.session_state['transcribed_text'] = ""

    st.session_state['transcribed_text'] = speech_to_text(key="audio_recorder")
       
    
    # Displaying the chat input
    user_input = st.chat_input("You (type your message here) or use start recording button above:", key="chat_input")

    # If there's transcribed audio, setting that as the user_input
    if st.session_state['transcribed_text']:
        user_input = st.session_state['transcribed_text']
        st.session_state['transcribed_text'] = ''  # Clearing the transcribed text after setting it to user_input

    if user_input:
        # Appending user input to session state chat history
        user_message = {"role": "user", "content": user_input}
        st.session_state["chat_history"].append(user_message)

        # Generatng response using the streamlit chatbot logic
        chatbot_response = get_chatbot_response(user_input, stuff_chain, db)
        chatbot_output_text = chatbot_response.get("output_text", "")

        # Appending chatbot response to session state chat history
        st.session_state["chat_history"].append({"role": "assistant", "content": chatbot_output_text})

        # Saving the updated chat history to ensure it is captured in case of relogging
        if 'email' in st.session_state:
            save_chat_history(st.session_state['email'], st.session_state['chat_history'])

        # Displaying the entire chat history including the latest response
        display_chat_history(collector)

def display_chat_history(collector):
    if "chat_history" in st.session_state:
        last_assistant_index = None
        # Determining the last index where the role is "assistant"
        for index, message in enumerate(st.session_state["chat_history"]):
            if message["role"] == "assistant":
                last_assistant_index = index

        # Displaying messages and manage feedback
        for index, message in enumerate(st.session_state["chat_history"]):
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                # This will write assistant messages as well
                st.chat_message("assistant").write(message["content"])

                # Only adding feedback for the last assistant message
                if message["role"] == "assistant" and index == last_assistant_index:
                    user_feedback=collector.st_feedback(
                        component="wealthwishper",
                        feedback_type="thumbs",
                        open_feedback_label="[Optional] Provide additional feedback",
                        model="gemini-1.0-pro-latest",
                        align="flex-start"
                        
                    )
                    print(user_feedback)



def save_chat_history(email, chat_history):
    try:
        client = storage.Client()
        bucket = client.bucket('credential_save')
        blob = bucket.blob(f"{email}.json")
        blob.upload_from_string(json.dumps(chat_history))
    except Exception as e:
        st.error(f"Failed to save chat history: {str(e)}")

def load_chat_history(email):
    """Loads the chat history from a Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.bucket('credential_save')
    blob = bucket.blob(f"{email}.json")
    try:
        data = blob.download_as_string()
        chat_history = json.loads(data)
        return chat_history
    except Exception as e:
        print(f"Failed to load chat history: {e}")
        return []  # Returning an empty list if no history exists

def fetch_top_headlines(keyword, category, language, country):
    """Fetches top headlines based on the specified parameters."""
    top_headlines = newsapi.get_top_headlines(q=keyword,
                                              category=category,
                                              language=language,
                                              country=country)
    return top_headlines['articles']

def display_news_in_sidebar():
    st.sidebar.title("Latest Finance News")
    keywords = ['Artificial Intelligence', 'investment', 'stock market','finance','government finance bills','budget']  # Multiple keywords
    all_articles = []

    for keyword in keywords:
        articles = fetch_top_headlines(keyword=keyword,
                                       category='business',
                                       language='en',
                                       country='in')
        all_articles.extend(articles)  # Combine articles from each keyword

    if all_articles:
        unique_articles = {art['title']: art for art in all_articles}.values()  # Remove duplicates based on title
        for article in unique_articles:
            st.sidebar.markdown(f"### {article['title']}")
            st.sidebar.info(article['description'])
            st.sidebar.markdown(f"[Read more]({article['url']})", unsafe_allow_html=True)
    else:
        st.sidebar.write("No news articles found.")

# Google OAuth flow setup
flow = Flow.from_client_secrets_file(
    'client_secrets.json',
    scopes=["openid", "https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email"],
    redirect_uri=REDIRECT_URI
)

def authenticate_user():
    st.title("💰Wealth Whisperer💰")
    authorization_url, state = flow.authorization_url()
    st.session_state['state'] = state
    # st.write(f"Please log in [here]({authorization_url})")
    if st.button('Log in'):
        webbrowser.open_new_tab(authorization_url)

def get_user_info(code):
    flow.fetch_token(code=code)
    credentials = flow.credentials
    session = requests.Session()
    user_info = session.get('https://www.googleapis.com/oauth2/v3/userinfo',
                            headers={'Authorization': f'Bearer {credentials.token}'}).json()
    return user_info

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

if st.session_state['authentication_status'] is None:
    if 'code' not in st.query_params:
        authenticate_user()
    else:
        code = st.query_params['code']
        user_info = get_user_info(code)
        if user_info:
            st.session_state['authentication_status'] = 'Authenticated'
            st.session_state['user_info'] = user_info
            if 'email' in user_info:
                st.session_state['email'] = user_info['email']
                # Loading the chat history now that we have the user's email
                st.session_state['chat_history'] = load_chat_history(user_info['email'])
            else:
                st.error("Email not available. Check your Google app permissions.")

if st.session_state['authentication_status'] == 'Authenticated':
    # "Redirecting" to the chat interface
    handle_chat_interaction(stuff_chain, db, collector)
    display_news_in_sidebar()
    if 'email' in st.session_state and 'chat_history_updated' not in st.session_state:
        save_chat_history(st.session_state['email'], st.session_state['chat_history'])
        st.session_state['chat_history_updated'] = True
        st.session_state['chat_history'] = load_chat_history(st.session_state['email'])
        display_chat_history(collector)


else:
    st.write("Please log in to access the chat service.")
