import streamlit as st
from yolov9.detect import run
import streamlit as st
import cv2
import numpy as np
from PIL import Image

def increase_resolution_in_place(image_path, target_width):
    """
    Увеличивает разрешение изображения до заданных размеров и сохраняет его в тот же файл.
    
    Параметры:
        image_path (str): Путь к изображению.
        target_width (int): Желаемая ширина изображения.
        target_height (int): Желаемая высота изображения.
    """
    # Открыть изображение
    image = Image.open(image_path)
    
    imgwidth=image.size[0]
    imgheight=image.size[1]
    
    targetheight = int(imgheight * (target_width / imgwidth))
    # Увеличить разрешение изображения
    high_res_image = image.resize((target_width, targetheight), Image.LANCZOS)
    
    # Перезаписать исходный файл увеличенным изображением
    high_res_image.save(image_path)



st.title("cosplay")
st.write("""
    загрузите изображение, которое вы хотите опознать
    """)

img = st.file_uploader("", type=["png", "jpg", "jpeg"])
if img is not None:
    img_pil = Image.open(img).convert("RGB")
    img_pil.save("img.jpg")
    increase_resolution_in_place("img.jpg", 1024)
    img_np = cv2.imread("img.jpg")[:, :, ::-1]
    st.image(img_np, use_column_width=True)

    out_path = run(source = "img.jpg",weights="sources/cosplay_detector.pt",conf_thres=0.25,iou_thres=0.3,agnostic_nms=True)
    st.image(open(str(out_path)+"/img.jpg", 'rb').read(), use_column_width=True)



