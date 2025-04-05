# QnA Chatbot
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv() # Take all the environment variables from .env file

import streamlit as st

# Load OpenAI model and get responses

def get_openai_response(question):
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),model="gpt-4o-2024-08-06", temperature=0.5)

    response = llm.invoke([{"role": "user", "content": question}])

    return response.content

# Initialize streamlit

st.set_page_config(page_title="QnA Demo")

st.header("Langchain Application")

# Capture input

input_text = st.text_input("Input: ", key="input")

# Button click handler

if input_text:
    response = get_openai_response(input_text)
    st.subheader("The response is:")
    st.write(response)
    