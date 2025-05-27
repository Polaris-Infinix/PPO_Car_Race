import torch 
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim 
from collections import deque

def tonumpy(tensor):
    tensor=tensor.to("cpu")
    tensor=tensor.detach()
    return tensor.numpy()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.actor=nn.Sequential(
            layer_init(nn.Linear(8,200)),
            nn.ReLU(),
            layer_init(nn.Linear(200,150)),
            nn.ReLU(),
            layer_init(nn.Linear(150,50)),
            nn.ReLU(),
            layer_init(nn.Linear(50,4)),
            nn.Softmax(dim=-1)
        )

        self.critic=nn.Sequential(
            layer_init(nn.Linear(8,200)),
            nn.ReLU(),
            layer_init(nn.Linear(200,150)),
            nn.ReLU(),
            layer_init(nn.Linear(150,50)),
            nn.ReLU(),
            layer_init(nn.Linear(50,1))
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=2.5e-4)
        self.critic_optimizer=optim.Adam(self.critic.parameters(),lr=1e-3)
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def get_action_and_value(self,state,action=None):
        state=torch.Tensor(state).to(device)
        act=self.actor(state)
        value=self.critic(state)
        dist=Categorical(probs=act)
        if action is None:
            action = dist.sample()
            log_prob=dist.log_prob(action)
            action = tonumpy(action)
            log_prob = tonumpy(log_prob)
            entropy=None

        else:
            log_prob = dist.log_prob(action)  
            action=action
            entropy = dist.entropy()
            # print(f'entropy{entropy}')

        return action, log_prob, value, entropy



class Memory(Network):
    def __init__(self, batch_size=5):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.batch_size = batch_size
        super(Memory,self).__init__()

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
    
        return np.array(self.states),\
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.done), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(tonumpy(vals))
        self.rewards.append(reward)
        self.done.append((done))

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.vals = []

    def give_only_reward(self):
        return sum(self.rewards)
    

    def advantages(self,gamma=0.99,lamda=0.95):
        values=self.vals.copy()
        rewards=self.rewards
        values.append(0)
        adv=deque(maxlen=len(rewards))
        for t in range(len(rewards)):
            delta=gamma*values[t+1]-values[t]+rewards[t]
            if adv:
                delta = adv[0] * gamma * lamda + delta       
            adv.appendleft(delta)
        adv=np.array(list(adv))
        return adv
    
    def learn(self):
        advantage=self.advantages()
        advantage=torch.tensor(advantage).to(device)
        for _ in range(8):
            state,action,prob,values,rewards,entropy,batches=self.generate_batches()
            values=torch.from_numpy(values).to(device)
        
            for batch in batches:
                states=torch.from_numpy(state[batch]).to(device)
                actions=torch.from_numpy(action[batch]).to(device)
                old_log_probs=torch.from_numpy(prob[batch]).to(device)
                _,new_log_prob,critic_value,entropy=self.get_action_and_value(states,actions)
                r_t=torch.exp(new_log_prob-old_log_probs)
                # print(r_t)
                weighted_probs=advantage[batch]*r_t
                clip_weighted_probs=advantage[batch]*torch.clamp(r_t,0.8,1.2)
                actor_loss=-torch.min(weighted_probs,clip_weighted_probs).mean()

                returns=advantage[batch]+values[batch]
                # print(f'critic before {critic_value}')
                critic_value=critic_value.squeeze(0)
                # print(f'critic {critic_value}')
                critic_loss=(returns-critic_value)**2
                critic_loss=critic_loss.mean()
                total_loss=actor_loss+0.5*critic_loss-0.01*entropy.mean()
                self.total_loss_wab=total_loss      
                self.returns=returns.mean()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.clear_memory()

    def save_model(self, filename="ppo_checkpoint.pth"):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            # "log_std": self.log_std,
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, "ppo_checkpoint.pth")

    def load_model(self, filename="ppo_checkpoint.pth"):
        checkpoint = torch.load(filename, map_location=device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.log_std = checkpoint["log_std"] 
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.to(device)



#idk what's wrong
#adding Entropy





        
        
        
        
        
        # print(states.shape)
        # print(values)
        # print(actions.shape)
        # print(actions)
        # print(probs.shape)
        # print(probs)
        # print(rewards.shape)
        # print(rewards)
        # print(batches) #Worked lol 




    