# Устанавливаем логгер для детектрона
import detectron2
# Для печати логов
from detectron2.utils.logger import setup_logger
# Инициализируем логгер
setup_logger()
from PIL import Image
# Импорты
import numpy as np
import os, json, cv2, random

# Зоопарк моделей (по аналогии с torchvision.models)
from detectron2 import model_zoo
# Отдельный класс для предикта разными моделями
from detectron2.engine import DefaultPredictor
# Всея конфиг: все будем делать через него
from detectron2.config import get_cfg

# Для визуализации
from detectron2.utils.visualizer import Visualizer

# Для собственного датасета
from detectron2.data import MetadataCatalog, DatasetCatalog

import streamlit as st
from detectron2.data.datasets import register_coco_instances


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


img = st.file_uploader("", type=["png", "jpg", "jpeg"])

if img is not None:
    img_pil = Image.open(img).convert("RGB")
    img_np = np.array(img_pil)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Преобразование изображения в формат OpenCV (BGR)
    
    st.image(img_np, use_column_width=True)
    out = predict(img_cv2)  # Предполагается, что функция predict() принимает изображение в формате OpenCV

    
    st.image(out, use_column_width=True)