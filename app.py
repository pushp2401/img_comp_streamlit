import streamlit as st
from PIL import Image
import os
import anthropic
import base64
from dotenv import load_dotenv
import cv2

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
# prompt = """
# You are analyzing two construction blueprint images (Image 1 and Image 2). For each image, you have:

# A set of objects (walls, doors, stairs, etc.)
# A set of “areas” (e.g., “Balcony,” “Living Room,” “Hallway,” “Bathroom,” etc.) 
# Task Requirements:
# Identify differences between Image 1 and Image 2:
# Newly added objects in Image 2 that were not in Image 1.
# Missing objects in Image 2 that were in Image 1.
# Objects that have changed location or have changed labels.
# Text or label changes, if available.
# For missing or newly added objects, describe their location in terms of relative position or known areas (not raw coordinates):
# For example, say “the missing doors were originally near the top-left corner, adjacent to the main hallway,” or “new walls have been added in the southeast corner, near the living room.”
# Avoid including numeric bounding boxes, polygon areas, or centers in the final explanation.
# If two objects (one in Image 1 and one in Image 2) have the same label and nearly identical centers, consider them the same object and do not report them as a difference.
# Whenever possible, use known area labels to describe positions (e.g., “within the dining area,” “just north of the bathroom,” “adjacent to the balcony,” etc.).
# Return a concise and correct Markdown summary with these differences, focusing on where changes occur.

# """
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
