from mmf.models.mmbt import MMBT
from torchray.utils import imsc
from torchvision import transforms
from multimodal_extremal_perturbation import *
from PIL import Image

def image2tensor(image_path):
    # convert image to torch tensor with shape (1 * 3 * 224 * 224)
    img = Image.open(image_path)
    p = transforms.Compose([transforms.Scale((224,224))])

    img,i = imsc(p(img),quiet=False)
    return torch.reshape(img, (1,3,224,224))

def torchray_multimodal_explain(image_path,text):
    #image_path = "static\\" + image_path
    model = MMBT.from_pretrained("mmbt.hateful_memes.images")
    model = model.to(torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"))

    image_tensor = image2tensor(image_path)

    image_tensor = image_tensor.to((torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")))

    mask_, hist_, output_tensor, summary, conclusion = multi_extremal_perturbation(
                                                                    model,
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
    Image = transforms.ToPILImage()(imsc(image_tensor[0],quiet=False)[0]).convert("RGB")

    Image.save("path/torchray.png")
    print(summary)
    return conclusion



