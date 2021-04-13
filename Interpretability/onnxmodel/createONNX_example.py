
from Printer import layer
import torch
import torch.nn as nn
import torch.nn.functional as F

import io
import numpy as np


import torch.onnx
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()


        # Define Neural Network layers
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = layer(hidden_dim,1)
        

        # initialize the weights and bias of model
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):

        # add activation functions and connect layers
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    def predict(self,state):
        logit = self.forward(state)

        return logit*1000

if __name__ == '__main__':
    torch_model = ValueNetwork(2,10)
    state = np.array([1.0,2.0])
    state = torch.FloatTensor(state)
    torch_model.eval()
    model_out = torch_model(state)

    print("original model ",model_out)
    print("model forward",torch_model.forward(state))
    print("model predict",torch_model.predict(state))

    torch.onnx.export(torch_model,               # model being run
                  state,                         # model input (or a tuple for multiple inputs)
                  "testModel.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}},
                  verbose=True)
