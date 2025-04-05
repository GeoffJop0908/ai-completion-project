import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

## Get response from LLama model

def get_gpt4o_response(input_text, no_words, blog_style):
    # Initialize the OpenAI GPT-4o model
    llm = ChatOpenAI(model="gpt-4o")

    # Prompt Template
    template = """
    Write a blog for {blog_style}. The topic is {input_text} and you should write this blog within {no_words} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template=template)

    # Generate response from GPT-4o
    response = llm.invoke(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    return response.content

st.set_page_config(
    page_title="Generate Blogs",
    page_icon='📝',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs 📝")

input_text = st.text_input("Enter the Blog Topic")

## Additional 2 fields

col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input('No. of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientists', 'General People', 'Other Bloggers', 'Market Professionals'), index = 0)

submit = st.button("Generate")

## Final response
if submit:
    st.write(get_gpt4o_response(input_text, no_words, blog_style))
