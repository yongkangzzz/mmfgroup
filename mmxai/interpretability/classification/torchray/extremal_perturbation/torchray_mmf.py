from mmxai.interpretability.classification.torchray.extremal_perturbation.multimodal_extremal_perturbation import *
from mmf.models.mmbt import MMBT
from mmf.models.fusions import LateFusion
from mmf.models.vilbert import ViLBERT
from mmf.models.visual_bert import VisualBERT
from PIL import Image


def torchray_multimodal_explain(image_name, text, model):
    print(image_name)
    print(text)
    image_path = "static/" + image_name

    image = Image.open(image_path)
    width, height = image.size

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
    PIL_image = transforms.ToPILImage()(
        imsc(image_tensor[0], quiet=False)[0]).convert("RGB")
    PIL_image = PIL_image.resize((width, height), Image.ANTIALIAS)

    name_split_list = image_name.split('.')
    exp_image = name_split_list[0] + '_torchray.' + name_split_list[1]
    PIL_image.save("static/" + exp_image)
    print(summary)
    return conclusion, exp_image
