import onnxruntime
import numpy as np
import torch
from torch import nn
import onnx
from onnx import helper, TensorProto, checker




class ONNXInterface:
	def __init__(self,model_path):
		self.model = onnx.load(model_path)
		self.ort_session = onnxruntime.InferenceSession(model_path)
		if not onnx.checker.check_model(self.model):
			assert("Model file error")
	def visualize(self):
		print(onnx.helper.printable_graph(self.model.graph))

	def onnx_model_forward(self, image_input,text_input):
		'''
		Args:
			image_input: the image torch.tensor with size (1,3,224,224)
			text_input : the text input Str
		Returns :
			logits computed by model.forward List()
		'''
	

		output_name = self.ort_session.get_outputs()[0].name
		input_name1 = self.ort_session.get_inputs()[0].name
		input_name2 = self.ort_session.get_inputs()[1].name

		image_input = to_numpy(image_input)

		ort_inputs = {input_name1: image_input,input_name2 :text_input}

		ort_outs = self.ort_session.run([output_name], ort_inputs)
		return ort_outs

	def classify(self,image_path,text_input, image_tensor = None):
		'''
		Args:	
			image_path: directory of input image
			text_input : the text input Str
			image_tensor : the image torch.tensor with size (1,3,224,224)
			
		Returns :
			label of model prediction and the corresponding confidence
		'''
		if image_tensor:
			logits = self.onnx_model_forward(image_tensor,text_input)
			scores = nn.functional.softmax(torch.tensor(logits), dim=1)
			return score
		else:
			image_tensor = image2tensor(image_path)
			logits = self.onnx_model_forward(image_tensor,text_input)
		scores = nn.functional.softmax(torch.tensor(logits), dim=1)
		confidence, label = torch.max(scores, dim=1)

		return {"label": label.item(), "confidence": confidence.item()}


def image2tensor(image_path):
    # convert image to torch tensor with shape (1 * 3 * 224 * 224)
    img = Image.open(image_path)
    p = transforms.Compose([transforms.Scale((224,224))])

    img,i = imsc(p(img),quiet=True)
    return torch.reshape(img, (1,3,224,224))



def to_numpy(tensor):
	return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

