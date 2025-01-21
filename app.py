import streamlit as st
from PIL import Image
import os
import anthropic
import base64
import numpy as np
from dotenv import load_dotenv
import cv2

load_dotenv() 


from yolo_functions import segment_large_image_with_tiles , usable_data , plot_differences_on_image1 , system_prompt_4
from ultralytics import YOLO 
from openai import OpenAI
import os



client = anthropic.Anthropic(
    # api_key="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",
    api_key = os.getenv('ANTHROPIC_API_KEY')
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 


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
        temperature=0,
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
    #################### yolo segment from here ################
    model = YOLO("best.pt") 

    final_output_1, class_mask_dict_1 = segment_large_image_with_tiles(
    model,
    # large_image_path=img_1_path,
    # large_image_path= tmp_img1 ,
    large_image_path= img1 
    tile_size=1080,
    overlap=120,
    alpha=0.4,
    display=True
    )
    final_output_2, class_mask_dict_2= segment_large_image_with_tiles(
    model,
    large_image_path=tmp_img2,
    tile_size=1080,
    alpha=0.4,
    display=True
    )
    label_dict = {0: 'EMP', 1: 'balcony_area', 2: 'bathroom', 3: 'brick_wall', 4: 'concrete_wall', 5: 'corridor', 6: 'dining_area', 7: 'door', 8: 'double_window', 9: 'dressing_room', 10: 'elevator', 11: 'elevator_hall', 12: 'emergency_exit', 13: 'empty_area', 14: 'lobby', 15: 'pantry', 16: 'porch', 17: 'primary_insulation', 18: 'rooms', 19: 'single_window', 20: 'stairs', 21: 'thin_wall'}
    img1_results = {}
    for key in class_mask_dict_1.keys():
        img1_results[label_dict[key]] = class_mask_dict_1[key]
    img2_results = {}
    for key in class_mask_dict_2.keys():
        img2_results[label_dict[key]] = class_mask_dict_2[key]
    image_1 , image_2 = img1 , img2 
    width, height = image_1.width, image_1.height 
    image_1_data = usable_data(img1_results, image_1) 
    image_2_data = usable_data(img2_results, image_2) 

    user_prompt_3 = f"""I have two construction blueprint images, Image 1 and Image 2, and here are their segmentation results (with bounding boxes, centers, and areas). Please compare them and provide a short Markdown summary of the differences, ignoring any objects that match in both images:
    
        Image 1:
        image: {image_1}
        
        json
        Copy
        {image_1_data}
        Image 2:
        image: {image_2}
        json
        Copy
        {image_2_data}
        
        Please:
        Compare the two images only in terms of differences—ignore any objects that match (same label and near-identical center).
        For objects missing in Image 2 (but present in Image 1), or newly added in Image 2, indicate their relative position using known areas or approximate directions. For instance, mention if the missing doors were “towards the north side, near the elevator,” or if new walls appeared “in the southeastern corner, near the balcony.”
        Summarize any changes in labels or text, again without giving raw bounding box or polygon coordinate data.
        Provide your final output in a short, clear Markdown summary that describes where objects have changed.
        Mention if there are text/label changes (e.g., from an OCR perspective) in any particular area or region
    """


    

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt_4},
            {
                "role": "user",
                "content": user_prompt_3
            }

        ]
    )

    print(completion.choices[0].message.content)






    
