from fastapi import FastAPI
import uvicorn
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
endpoint_url = (
    "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
)

# Defining the LLM model
llm = HuggingFaceEndpoint(endpoint_url=endpoint_url)

# Defining the prompt templates
# promptForGeography = ChatPromptTemplate.from_messages({
#     "system": "You are a helpful assistant who answers geography related questions. Please response to the user queries",
#     "user": "Question:{question}"
# })

# promptForCode = ChatPromptTemplate.from_messages({
#     "system": "You are a helpful assistant who answers code related questions. Please response to the user queries",
#     "user": "Question:{question}"
# })

promptForGeography = ChatPromptTemplate.from_template("Please response to this question related to geography {question}")

promptForCode = ChatPromptTemplate.from_template("Please response to this question related to code {question}")


# Creating a FastAPI app
app = FastAPI(
    title = "Langchain API with HuggingFace",
    description = "API to interact with Langchain using HuggingFace",
    version="1.0"
)

# Adding the routes to the FastAPI app
add_routes(
    app,
    promptForGeography|llm,
    path = "/geography"
)

add_routes(
    app,
    promptForCode|llm,
    path = "/code"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
