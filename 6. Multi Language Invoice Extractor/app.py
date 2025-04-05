from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
model = 'gemini-2.0-flash-001'

# Get Gemini's response
def get_gemini_response(input, image, prompt):
    contents = [
        {
            "role": "user",
            "parts": [{"text": input}]
        },
        {
            "role": "user",
            "parts": image
        },
        {
            "role": "user",
            "parts": [{"text": prompt}]
        }
    ]
    
    response = client.models.generate_content(
        model=model,
        contents=contents
    )
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        image_parts = [{
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": bytes_data
            }
        }]

        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")

# Streamlit setup

st.set_page_config(page_title="Multilanguage Invoice Extractor")

st.header("Multilanguage Invoice Extractor")
input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an image of the invoice:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

submit = st.button("Tell me about the invoice")

input_prompt = """
You are an expert in understanding invoices. We will upload an image of an invoice and you must answer any questions based on this uploaded invoice image.
"""

# Submit button click
if submit:
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input)

    st.subheader("The response:")
    st.write(response)