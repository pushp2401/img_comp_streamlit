import streamlit as st
from PIL import Image
import os
import anthropic
import base64
from dotenv import load_dotenv

load_dotenv() 
client = anthropic.Anthropic(
    api_key= os.getenv('ANTHROPIC_API_KEY'),
)


def encode_image(image_path):
    # with open(image_path, "rb") as image_file:
        # return base64.b64encode(image_file.read()).decode("utf-8")
    return base64.b64encode(image_path.getvalue()).decode("utf-8")

def chat_claude(prompt , image1 , image2 ) :
    # print("image 1"  , image1)
    image1_data = encode_image(image1) 
    # print("image 1 data" , image1_data)
    image2_data = encode_image(image2) 
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens = 4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Image 1:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image1_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Image 2:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image2_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"{prompt}"
                    }
                ],
            }
        ],
    )
    return message.content[0].text


prompt = """Given 2 construction blueprints your task is to analyze carefully both blueprints and point out difference for following categories -
1. Strcutural grid.
2. Layout Areas - rooms , balcony , porch , staircase , elevator etc.
3. Interior changes or optimization.
Summarize all the difference in paragraph concisely.
"""
st.set_page_config(layout = "wide")
uploaded_files = st.file_uploader("Upload 2 image to compare", accept_multiple_files=True)
# import pdb; pdb.set_trace()
# print("upladed file length" , len(uploaded_files))
if len(uploaded_files) !=0  :
    i = 0
    for one_file in uploaded_files :
        if i == 0 :
            img1 = Image.open(one_file)  
            tmp_img1 = one_file    
            st.image(img1)
            i = i + 1
        if i == 1 :
            img2 = Image.open(one_file) 
            tmp_img2 = one_file     
            st.image(img2)
            i = i + 1

    col1 , col2 = st.columns(2)
    col1.header("LLM")
    col2.header("Seg-LLM !")
    # import pdb; pdb.set_trace()
    llm_ans = chat_claude(prompt , tmp_img1 , tmp_img2)
    print(llm_ans)
    col1.write(llm_ans)
