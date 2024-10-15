from langchain_community.llms import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import streamlit as st
import os

# Loading all the constants and environment variables
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
endpoint_url = (
    "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
)

# Defining the LLM model
llm = HuggingFaceEndpoint(endpoint_url=endpoint_url)

# Defining the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}"),
    ]
)

# Defining the output parser
outputParser = StrOutputParser()

# creating a chain
chain = prompt | llm | outputParser

# Streamlit Framework
st.title("Using Langchain with HuggingFace MistralAi-7B")
input_text = st.text_input("Enter the prompt")
if input_text:
    st.write(chain.invoke({"question": input_text}))
