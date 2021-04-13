from mmf.models.mmbt import MMBT
from mmf.models.fusions import LateFusion
from mmf.models.vilbert import ViLBERT
from mmf.models.visual_bert import VisualBERT
import torch
from shap4mmf._explainer import Explainer
import numpy as np
from PIL import Image
import os


def shap_multimodal_explain(image_name, text, model):
    image = "static/" + image_name
    output = model.classify(image, text)
    # Explainer hyper params
    max_evals = 100
    batch_size = 50
    explainer = Explainer(model, max_evals=max_evals, batch_size=batch_size)

    target_images = Image.open(image)
    target_images = np.array(target_images, dtype=np.uint8)
    if target_images.shape[2] > 3:
        target_images = target_images[:, :, :3]
    target_images = target_images.reshape(1, target_images.shape[0], target_images.shape[1], target_images.shape[2])
    target_texts = np.array([text])
    image_shap_values, text_shap_values = explainer.explain(target_images, target_texts, "multimodal")
    PIL_image_list = explainer.image_plot(image_shap_values)
    text_exp_list = explainer.parse_text_values(text_shap_values, 1)
    PIL_image = PIL_image_list[0]

    name_split_list = os.path.splitext(image_name)
    exp_image = name_split_list[0] + '_shap.png'
    PIL_image.save("static/" + exp_image)

    return text_exp_list[0], exp_image
