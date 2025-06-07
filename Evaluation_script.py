import gymnasium as gym 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.distributions import Categorical


env=gym.make("LunarLander-v3", render_mode="human")
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class agent(nn.Module):
    def __init__(self):
        super(agent,self).__init__()
        self.actor=nn.Sequential(
            nn.Linear(8,50),
            nn.ReLU(),
            nn.Linear(50,112),
            nn.ReLU(),
            nn.Linear(112,152),
            nn.ReLU(),
            nn.Linear(152,50),
            nn.ReLU(),
            nn.Linear(50,4)
        )
        self.critic=nn.Sequential(
            nn.Linear(8,50),
            nn.ReLU(),
            nn.Linear(50,112),
            nn.ReLU(),
            nn.Linear(112,152),
            nn.ReLU(),
            nn.Linear(152,50),
            nn.ReLU(),
            nn.Linear(50,1)
        )
    
    def values(self,obs):
        return self.critic(obs)
    
    def get_actions_probs(self,state,action=None):
        dist=Categorical(logits=self.actor(state))
        value=self.critic(state)
        if action is None:
            action=dist.sample()
        log_prob=dist.log_prob(action)
        entropy=dist.entropy()
        return action, log_prob,entropy,value 
    
model=agent().to(device)
checkpoint = torch.load("ppo.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # Set to evaluation mode if not training
    
while True:
    state, _ = env.reset()
    rewards=[]
    done = False
    while not done:
        state_tensor = torch.tensor(state).unsqueeze(0).float().to(device)
        with torch.no_grad():
            action, _, _, _ = model.get_actions_probs(state_tensor)
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated
    print(f'Reward of an episode {sum(rewards)}')

