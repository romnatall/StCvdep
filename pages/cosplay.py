from urllib import response
import streamlit as st
from yolov9.detect import run
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO



def increase_resolution_in_place(image_path, target_width):
    image = Image.open(image_path)
    imgwidth=image.size[0]
    imgheight=image.size[1]
    targetheight = int(imgheight * (target_width / imgwidth))
    high_res_image = image.resize((target_width, targetheight), Image.LANCZOS)
    high_res_image.save(image_path)



st.title("cosplay")
st.write("""
    загрузите изображение, которое вы хотите опознать
    """)
uploaded_files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
url = st.text_input("Enter image url")

def pred(img):
    img_pil = Image.open(img).convert("RGB")
    img_pil.save("img.jpg")
    increase_resolution_in_place("img.jpg", 1024)
    img_np = cv2.imread("img.jpg")[:, :, ::-1]
    st.image(img_np, use_column_width=True)
    out_path = run(source = "img.jpg",weights="sources/cosplay_detector.pt",conf_thres=0.25,iou_thres=0.3,agnostic_nms=True)
    st.image(open(str(out_path)+"/img.jpg", 'rb').read(), use_column_width=True)

if url:  # Check if URL is not empty
    try:
        response = requests.get(url)
        if response.status_code == 200:  # Check if response status code is 200 (OK)
            img = BytesIO(response.content)  # Open the image from the response content
            pred(img)
        else:
            st.warning("Invalid URL. Make sure the URL is correct and the image exists.")
        
        # Process the response further if needed
    except requests.exceptions.MissingSchema:
        st.warning("Invalid URL format. Make sure to include 'http://' or 'https://'")

if uploaded_files is not None:
        for img in uploaded_files:
            pred(img)






