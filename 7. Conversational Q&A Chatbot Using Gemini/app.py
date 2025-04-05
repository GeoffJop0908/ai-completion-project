from dotenv import load_dotenv
import streamlit as st
import os
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
model = 'gemini-2.5-pro-exp-03-25'

## Get gemini response
chat = client.chats.create(model=model)

def get_gemini_response(query):
    return chat.send_message_stream(query)

st.set_page_config(page_title="Q&A Demo with Gemini")

st.header("Chat with Gemini")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

input = st.text_input("Input:", key="input")
submit = st.button("Ask the question")

if submit and input:
   response = get_gemini_response(input)

   st.session_state['chat_history'].append(("You", input))
   st.subheader("Response:")

   accumulated_text = ""

   response_placeholder = st.empty()

   for chunk in response:
    accumulated_text += chunk.text
    response_placeholder.markdown(accumulated_text)
       
   st.session_state['chat_history'].append(("Bot", accumulated_text))

st.subheader("Chat History")

for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")