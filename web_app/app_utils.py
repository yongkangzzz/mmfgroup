from mmf.models.mmbt import MMBT
from mmf.models.fusions import LateFusion
from mmf.models.vilbert import ViLBERT
from mmf.models.visual_bert import VisualBERT
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def prepare_explanation(img_name, img_text, user_model, model_type, model_path, encourage):
    if model_path is not None:
        model_path = 'static/' + model_path
    img_name = 'static/' + img_name

    model = setup_model(user_model, model_type, model_path)
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    img = check_image(img_name)
    output = model.classify(img, img_text)
    
    if encourage == "encourage":
        label_to_explain = output["label"]
    else:
        label_to_explain = abs(output["label"] - 1)
    label = output["label"]
    conf = output["confidence"]
    
    return model, label_to_explain, label, conf
    

def setup_model(user_model, model_type, model_path):
    if user_model == "no_model":
        if model_type == "MMBT":
            model = MMBT.from_pretrained("mmbt.hateful_memes.images")
        elif model_type == "LateFusion":
            model = LateFusion.from_pretrained("late_fusion.hateful_memes")
        elif model_type == "ViLBERT":
            model = ViLBERT.from_pretrained("vilbert.finetuned.hateful_memes.direct")
        else:   # visual bert
            model = VisualBERT.from_pretrained("visual_bert.finetuned.hateful_memes.direct")
            
    elif user_model == "mmf":
        try:
            if model_type == "MMBT":
                model = MMBT.from_pretrained(model_path)
            elif model_type == "LateFusion":
                model = LateFusion.from_pretrained(model_path)
            elif model_type == "ViLBERT":
                model = ViLBERT.from_pretrained(model_path)
            else:
                model = VisualBERT.from_pretrained(model_path)
        except:
            return "Sorry, we cannot open the mmf checkpoint you uploaded. It should be an .ckpt file saved from the mmf trainer."

    else:
        model = MMBT.from_pretrained("mmbt.hateful_memes.images")
    # elif user_model == "onnx": ?????
    return model


def check_image(image_name):
    try:
        img = Image.open(image_name)
    except:
        return "Sorry, we cannot open the image file you uploaded"
    return img


def text_visualisation(exp, pred_res, save_path):
    if pred_res == 1:
        plt_title = "hateful"
    else:
        plt_title = "not hateful"

    print(exp)

    # handle different output formats from explainers
    vals = []
    names = []
    if isinstance(exp, dict):
        for i in exp:
            names.append(i)
            vals.append(exp[i])
    elif isinstance(exp, list) and isinstance(exp[0], str):
        for i in exp:
            names.append(i.split()[0])
            vals.append(float(i.split()[-1]))
    elif isinstance(exp, list) and isinstance(exp[0], list):
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]

    vals.reverse()
    names.reverse()

    if len(vals) <= 2:
        fig = plt.figure(figsize=(6, 2))
    else:
        fig = plt.figure()

    colors = ['hotpink' if x > 0 else 'cornflowerblue' for x in vals]
    pos = np.arange(len(exp)) + .5
    plt.barh(pos, vals, align='center', color=colors)
    plt.yticks(pos, names)
    title = 'Contribution of each word to your model prediction: ' + plt_title
    plt.title(title)
    plt.savefig('static/' + save_path, transparent=True)
