
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from PIL import Image
import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import streamlit as st
from detectron2.data.datasets import register_coco_instances
import requests
from io import BytesIO


try:
    style_valid_meta = MetadataCatalog.get("style_valid")
except:
    register_coco_instances(
        name="style_valid",
        metadata={},
        json_file="sources/_annotations.coco.json",
        image_root="")
    style_valid_meta = MetadataCatalog.get("style_valid")

@st.cache_resource
def get_predictor():

    cfg = get_cfg()
    seg_model = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(seg_model))
    #cfg.MODEL.WEIGHTS = "sources/fashion_detector.pth"  # Относительный путь к файлу весов
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(seg_model)
    cfg.MODEL.DEVICE = "cpu" 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    return predictor

predictor = get_predictor()


def predict(im):
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    metadata=style_valid_meta,
                    scale=0.8
        )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #out = v.overlay_instances(masks = outputs["instances"].pred_masks.to("cpu")) 
    return out.get_image()#[:, :, ::-1]


st.title("Fashion")
st.write("""
    загрузите изображение, которое вы хотите отобразить (файл с моей моделью закорраптился, так что тут стандартная, когда-нибудь заменю)
    """)
uploaded_files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
url = st.text_input("Enter image url")

def pred(img):
    img_pil = Image.open(img).convert("RGB")
    img_np = np.array(img_pil)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  
    st.image(img_np, use_column_width=True)
    out = predict(img_cv2) 
    st.image(out, use_column_width=True)


if url: 
    try:
        response = requests.get(url)
        if response.status_code == 200: 
            img = BytesIO(response.content)  
            pred(img)
        else:
            st.warning("Invalid URL. Make sure the URL is correct and the image exists.")
    except requests.exceptions.MissingSchema:
        st.warning("Invalid URL format. Make sure to include 'http://' or 'https://'")

if uploaded_files is not None:
        for img in uploaded_files:
            pred(img)

