from mmxai.interpretability.classification.lime.lime_multimodal import LimeMultimodalExplainer
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
import numpy as np
from PIL import Image
import torch
from mmf.models.mmbt import MMBT
from mmf.models.fusions import LateFusion
from mmf.models.vilbert import ViLBERT
from mmf.models.visual_bert import VisualBERT

def lime_multimodal_explain(image_path, text, model, label_to_exp):
    image = 'static/' + image_path
    img_try = Image.open(image)
    text = text
    image_numpy = np.array(img_try)

    def multi_predict(model, imgs, txts, zero_image=False, zero_text=False):
        inputs = zip(imgs, txts)
        res = np.zeros((len(imgs), 2))
        for i, this_input in enumerate(inputs):
            img = Image.fromarray(this_input[0])
            txt = this_input[1]
            this_output = model.classify(img, txt, zero_image=zero_image, zero_text=zero_text)
            res[i][this_output["label"]] = this_output["confidence"]
            res[i][1 - this_output["label"]] = 1 - this_output["confidence"]
        return res
    
    exp1 = LimeMultimodalExplainer(image_numpy, text, model)
    explanation1 = exp1.explain_instance(multi_predict, 500)
    _, _, text_list, temp, mask = explanation1.get_explanation(label_to_exp)
    img_boundry = mark_boundaries(temp, mask)
    img_boundry = img_as_ubyte(img_boundry)
    PIL_image = Image.fromarray(np.uint8(img_boundry)).convert('RGB')

    name_split_list = image_path.split('.')
    exp_image = name_split_list[0] + '_lime.' + name_split_list[1]
    PIL_image.save("static/" + exp_image)

    text_exp_list = []
    for pair in text_list:
        text_exp_list.append(list(pair))

    return text_exp_list, exp_image
