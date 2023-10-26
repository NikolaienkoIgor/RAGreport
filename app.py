# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

st.markdown("""
    <style>
    .stApp {
        background-color: black;
    }
    div[data-baseweb="input"] > div {
        background-color: white;
        color: black;
    }
    .stMarkdown {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Colorful title
st.markdown("""
    <h1><span style='color: red;'>my</span><span style='color: yellow;'>Company</span><span style='color: pink;'>Brain🧠</span><span style='color: pink;'>Tank</span></h1>
    """, unsafe_allow_html=True)

# Set APIkey for OpenAI Service
with open("api_key.txt", "r") as file:
    api_key = file.read().strip()
os.environ['OPENAI_API_KEY'] = api_key

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Create and load PDF Loader
loader = PyPDFLoader('Report2023.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='Report2023')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="finance_report",
    description="a finance report as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# Create a text input box for the user
prompt = st.text_input('Input your request here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # # With a streamlit expander  
    # with st.expander('Document Similarity Search'):
    #     # Find the relevant pages
    #     search = store.similarity_search_with_score(prompt) 
    #     # Write out the first 
    #     st.write(search[0][0].page_content)