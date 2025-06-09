import gymnasium as gym 
import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.distributions import Categorical
import wandb
from  torch.distributions import Normal
from environment_handler import Environment

# env = gym.make("LunarLander-v3")
env=Environment()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalize = True
learning_rate = 3e-4
mini_batch = 64
batch_size = 2048
epoch = 10
clipping_eps = 0.2
wb=False
if wb:
    wandb.init(
        project="Car_racing",          
        name="run-001",              
        config={
            "act_learning_rate": learning_rate,
            "critic_learning_rate":learning_rate,
            "batch_size": batch_size,
            "update_epochs": 10,
            "clip_coef": 0.2,
            "env": "Lunar_lander",
            "Normalize":normalize,
            "mini batch size":mini_batch,
            "done":0
        }
    )
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class agent(nn.Module):
    def __init__(self):
        super(agent,self).__init__()
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
            layer_init(nn.Linear(512,200)),
            nn.ReLU(),
            layer_init(nn.Linear(200,3))

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
            layer_init(nn.Linear(512,1))
        )
    
    @torch.no_grad()
    def act(self, obs):
        mean=self.actor(obs)
        std=torch.exp(self.log_std)
        dist=Normal(mean,std)
        raction = dist.rsample().squeeze(0)
        log_prob=dist.log_prob(raction).squeeze(0)
        raction_0 = raction[0:1]
        raction_1 = raction[1:3]
        print(log_prob[0:1],log_prob[1:3])
        action_0 = torch.tanh(raction_0)
        action_1 = torch.sigmoid(raction_1)
        print(action_0, action_1)
        action = torch.cat([action_0, action_1], dim=-1) 
        log_prob_0 = log_prob[0:1] - torch.log(1 - action_0.pow(2) + 1e-6)
        log_prob_1 = log_prob[1:3] - torch.log(action_1 * (1 - action_1) + 1e-6)
        log_prob = (log_prob_0.sum() + log_prob_1.sum())
        value = self.critic(obs)
        print(action,log_prob,value)
        return action, log_prob, value
    
    def get_actions_probs(self, obs, action):
        mean=self.actor(obs)
        std=torch.exp(self.log_std)
        dist=Normal(mean,std)
        action_0 = action[..., 0:1] 
        action_1 = action[..., 1:3] 
        log_prob = dist.log_prob(action)  
        log_prob_0 = log_prob[..., 0:1] - torch.log(1 - action_0.pow(2) + 1e-6)
        log_prob_1 = log_prob[..., 1:3] - torch.log(action_1 * (1 - action_1) + 1e-6)
        log_prob = torch.cat([log_prob_0, log_prob_1], dim=-1).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        value = self.critic(obs)

        return log_prob, entropy, value

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(rewards.size(0))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * last_gae * mask
        advantages[t] = last_gae
        next_value = values[t]
    
    returns = advantages + values
    return advantages, returns

model = agent().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5) 

state = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device)

for gen in range(2000):
    #Tensors for storage
    states = torch.zeros(batch_size,3,96,288).to(device)
    actions = torch.zeros(batch_size,3).to(device)
    rewards = torch.zeros(batch_size, 1).to(device)
    values = torch.zeros(batch_size, 1).to(device)
    log_probs = torch.zeros(batch_size, 1).to(device)
    dones = torch.zeros(batch_size, 1).to(device)
    
    for t in range(batch_size):
        with torch.no_grad():
            action, log_prob, value = model.act(state.unsqueeze(0))
        next_state, reward, terminated, truncated = env.input(np.array([action[0].item(),action[1].item(),action[2].item()]))
        done = terminated or truncated
        print(t)
        states[t] = state
        print(states.size())
        actions[t] = action
        rewards[t] = reward
        values[t] = value
        log_probs[t] = log_prob
        dones[t] = float(done)
       
        state = torch.tensor(next_state, dtype=torch.float32, device=device)

        if done:
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

    with torch.no_grad():
        next_val = model.critic(state.unsqueeze(0))

    advantages, returns = compute_gae(rewards, values, dones, next_val)
    
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for _ in range(epoch):
        indices = torch.randperm(batch_size, device=device)
        
        for start in range(0, batch_size, mini_batch):
            batch_indices = indices[start:start+mini_batch]
            print("Hello")
            state_batch = states[batch_indices]
            action_batch = actions[batch_indices]
            log_prob_batch = log_probs[batch_indices]
            advantage_batch = advantages[batch_indices]
            return_batch = returns[batch_indices]

            new_probs, entropy, value = model.get_actions_probs(state_batch, action_batch)
            ratio = torch.exp(new_probs - log_prob_batch)
            clipped_ratio = torch.clamp(ratio, 1 - clipping_eps, 1 + clipping_eps)

            actor_loss = -torch.min(ratio * advantage_batch, clipped_ratio * advantage_batch).mean()
            
            critic_loss = (return_batch - value).pow(2).mean()
            
            entropy_loss = entropy.mean()

            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


    print(f"Gen: {gen:4d} | Returns: {returns.mean():6.0f} | Rewards: {rewards.sum():4.0f} | " +
          f"Actor loss: {actor_loss:0.2f} | Critic loss: {critic_loss:0.2f} | " +
          f"Entropy: {entropy_loss:1.2f}")
    if wb:
        wandb.log({
        "returns":returns.mean(),
        "reward":rewards.sum(),
        # "total_loss" : sum(tl) / len(tl),
        "Actor_loss":actor_loss,
        "critic_loss":critic_loss,
        "Entropy":entropy_loss   

})
