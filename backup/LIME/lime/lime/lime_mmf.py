from lime_multimodal import *
from skimage.segmentation import mark_boundaries
import numpy as np
from PIL import Image

from mmf.models.mmbt import MMBT
from mmf.models.fusions import LateFusion
from mmf.models.vilbert import ViLBERT
from mmf.models.visual_bert import VisualBERT


def setup_model(user_model, model_type, model_path):

    if user_model == "no_model":
        if model_type == "mmbt":
            model = MMBT.from_pretrained("mmbt.hateful_memes.images")
        elif model_type == "fusion":
            fusion = LateFusion.from_pretrained("late_fusion.hateful_memes")
        elif model_type == "vilbert":
            vilbert = ViLBERT.from_pretrained("vilbert.finetuned.hateful_memes.direct")
        elif model_type == "visual_bert"
            visual_bert_model = VisualBERT.from_pretrained("visual_bert.finetuned.hateful_memes.direct")
            
    elif user_model == "mmf":
        if model_type == "mmbt":
            model = MMBT.from_pretrained(model_path)
        elif model_type == "fusion":
            fusion = LateFusion.from_pretrained(model_path)
        elif model_type == "vilbert":
            vilbert = ViLBERT.from_pretrained(model_path)
        elif model_type == "visual_bert"
            visual_bert_model = VisualBERT.from_pretrained(model_path)
            
    # elif user_model == "onnx": ?????
    return model


def lime_multimodal_explain(image_path, text, user_model, model_type, model_path):
    model = setup_model(user_model, model_type, model_path)
    image_path = image_path
    img_try = Image.open(image_path)
    text = text
    image_numpy = np.array(img_try)

    def multi_predict(model, imgs, txts, zero_image=False, zero_text=False):
        inputs = zip(imgs, txts)
        res = np.zeros((len(imgs), 2))
        for i, this_input in enumerate(inputs):
            img = Image.fromarray(this_input[0])
            txt = this_input[1]
            this_output = model.classify(img, txt, zero_image, zero_text)
            res[i][this_output["label"]] = this_output["confidence"]
            res[i][1 - this_output["label"]] = 1 - this_output["confidence"]
        return res
    
    exp1 = LimeMultimodalExplainer(image_numpy, text)
    explanation1 = exp1.explain_instance(multi_predict, 250)
    list = explanation1.as_list()
    temp, mask = explanation1.get_image_and_mask(explanation1.top_labels[0], positive_only=True, num_features=3,
                                                 hide_rest=False)
    img_boundry = mark_boundaries(temp, mask)
    return list, img_boundry
