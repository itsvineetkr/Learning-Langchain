import requests
import streamlit as st


def get_geography_response(prompt):
    url = "http://localhost:8000/geography/invoke"
    payload = {"input": {"question": prompt}, "config": {}, "kwargs": {}}
    response = requests.post(url=url, json=payload)
    return response.json()['output']


def get_code_response(prompt):
    url = "http://localhost:8000/code/invoke"
    payload = {"input": {"question": prompt}, "config": {}, "kwargs": {}}
    response = requests.post(url=url, json=payload)
    return response.json()['output']


st.title("Using Langserve API with HuggingFace MistralAi-7B")
geographyQuestion = st.text_input("Enter the geography question")
codeQuestion = st.text_input("Enter the code question")

if geographyQuestion and not codeQuestion:
    st.write(get_geography_response(geographyQuestion))

if codeQuestion and not geographyQuestion:
    st.write(get_code_response(codeQuestion))
