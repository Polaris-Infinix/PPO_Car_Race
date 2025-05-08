import torch 
import numpy as np
import torch.nn as nn

# Layer initialization 
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.actor=nn.Sequential(
            layer_init(nn.Conv2d(3,16,5,padding=1,stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16,32,4,2,1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32,64,3,2,1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.linear(27648,512)),
            nn.ReLU,

        
)

    def forward(self,image):
        print(self.actor(image.unsqueeze(0)).size())

