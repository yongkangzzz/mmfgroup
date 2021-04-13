from torchray4mmf.multimodal_extremal_perturbation import *
from mmf.models.mmbt import MMBT
from mmf.models.fusions import LateFusion
from mmf.models.vilbert import ViLBERT
from mmf.models.visual_bert import VisualBERT


def setup_model(user_model, model_type, model_path):

    if user_model == "no_model":
        if model_type == "mmbt":
            model = MMBT.from_pretrained("mmbt.hateful_memes.images")
        elif model_type == "fusion":
            model = LateFusion.from_pretrained("late_fusion.hateful_memes")
        elif model_type == "vilbert":
            model = ViLBERT.from_pretrained("vilbert.finetuned.hateful_memes.direct")
        else:   # visual bert
            model = VisualBERT.from_pretrained("visual_bert.finetuned.hateful_memes.direct")
            
    elif user_model == "mmf":
        if model_type == "mmbt":
            model = MMBT.from_pretrained(model_path)
        elif model_type == "fusion":
            model = LateFusion.from_pretrained(model_path)
        elif model_type == "vilbert":
            model = ViLBERT.from_pretrained(model_path)
        else:
            model = VisualBERT.from_pretrained(model_path)

    else:
        model = MMBT.from_pretrained("mmbt.hateful_memes.images")
    # elif user_model == "onnx": ?????
    return model


def torchray_multimodal_explain(image_name, text, user_model, model_type, model_path):
    print(image_name)
    print(text)
    print(user_model)
    print(model_type)
    print(model_path)
    image_path = "static/" + image_name
    model = setup_model(user_model, model_type, model_path)
    model = model.to(torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"))

    image_tensor = image2tensor(image_path)

    image_tensor = image_tensor.to((torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")))

    mask_, hist_, output_tensor, summary, conclusion = multi_extremal_perturbation(model,
                                                                                   image_tensor,
                                                                                   image_path,
                                                                                   text,
                                                                                   0,
                                                                                   reward_func=contrastive_reward,
                                                                                   debug=True,
                                                                                   areas=[0.12])
    # summary is a higher level explanation in terms of sentence
    # conclusion is a list that contains words and their weights
    # output_tensor is the masked image
    image_tensor = output_tensor.to("cpu")
    PIL_image = transforms.ToPILImage()(imsc(image_tensor[0], quiet=False)[0]).convert("RGB")

    name_split_list = image_name.split('.')
    exp_image = name_split_list[0] + '_torchray.' + name_split_list[1]
    PIL_image.save("static/" + exp_image)
    print(summary)
    return conclusion, exp_image
