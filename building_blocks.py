import torch 
import numpy as np
import torch.nn as nn
from  torch.distributions import Normal

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
            layer_init(nn.Linear(27648,512)),
            nn.ReLU(),
            layer_init(nn.Linear(512,3)),
        )
        self.log_std=nn.Parameter(torch.zeros(1,3))
        self.critic=nn.Sequential(
            layer_init(nn.Conv2d(3,16,5,padding=1,stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16,32,4,2,1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32,64,3,2,1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(27648,512)),
            nn.ReLU(),
            layer_init(nn.Linear(512,1)),
        )

    def get_action_and_value(self,state):
        mean=self.actor(state)
        value=self.critic(state)
        std=torch.exp(self.log_std)
        dist=Normal(mean,std)
        print(mean,std)
        raction=dist.rsample()
        action=torch.tanh(raction)
        print(action)
        log_prob = dist.log_prob(raction) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob, value


        # value=self.critic(state)
        # mean=self.actor(state)
        # action_logstd=mean
        # action_std=torch.exp(action_logstd)
        # probs=Normal(mean,action_std)
        # action=probs.sample()
        # print(probs.log_prob(action),action)



    

    def forward(self,image):
        print(self.actor(image.unsqueeze(0)).size())

