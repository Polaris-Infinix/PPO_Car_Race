import torch 
import numpy as np
import torch.nn as nn
from  torch.distributions import Normal
import torch.optim as optim 
from collections import deque

def tonumpy(tensor):
    tensor=tensor.to("cpu")
    tensor=tensor.detach()
    return tensor.numpy()
# Layer initialization 

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        self.actor_optimizer=optim.Adam(self.actor.parameters(),lr=2.5e-4)
        self.critic_optimizer=optim.Adam(self.critic.parameters(),lr=1e-3)
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def get_action_and_value(self,state,action=None):
        mean=self.actor(state)
        value=self.critic(state)
        std=torch.exp(self.log_std)
        dist=Normal(mean,std)

        if action is None:
            raction=dist.rsample().squeeze(0)
            raction_0 = raction[0:1]  
            raction_1 = raction[1:3]


            action_0 = torch.tanh(raction_0)
            action_1 = torch.sigmoid(raction_1)
            action = torch.cat((action_0, action_1), dim=-1)  
        
            print(dist.log_prob(raction))
            log_prob_0 = dist.log_prob(raction).squeeze()[0] - torch.log(1 - action_0.pow(2) + 1e-6)
            log_prob_1 = dist.log_prob(raction).squeeze()[1] - torch.log(action_1[0] * (1 - action_1[0]) + 1e-6)
            log_prob_2 = dist.log_prob(raction).squeeze()[2] - torch.log(action_1[1] * (1 - action_1[1]) + 1e-6)
            print(log_prob_0)
            print(log_prob_1)
            print(log_prob_2)

            log_prob = log_prob_0 + log_prob_1 + log_prob_2
            print("log")
            action=torch.cat((action_0,action_1),dim=-1)
            entropy=dist.entropy().sum(dim=-1)
            action=tonumpy(action)
            log_prob=tonumpy(log_prob.squeeze())
            print(log_prob)
            print(f"No Grad strings-{action}")

        else:
           print("Given action")
           print(action)
           raction=dist.rsample().squeeze(0)
           print(raction)
           raction_0 = raction[...,0:1]  
           print(raction_0)
           raction_1 = raction[...,1:3]  
           print(raction_1)
           action_0=action[...,0:1]
           action_1=action[...,1:3]
           print("actions")
           print(action_0,action_1)


           
           log_prob_0 = dist.log_prob(raction)[...,0:1] - torch.log(1 - action_0.pow(2) + 1e-6)
           log_prob_1 = dist.log_prob(raction)[...,1:3] - torch.log(action_1 * (1 - action_1) + 1e-6)
           join = torch.cat((log_prob_0,log_prob_1),dim=1)
           log_prob=join.sum(1)
           
           print(log_prob_0,log_prob_1)
           print(log_prob)
           entropy=dist.entropy().squeeze()
           print(f"Grad_actions and logs snad entropy{action} and entropy is {entropy}")
           exit()

        
        return action, log_prob,value,entropy
    

class Memory(Network):

    def __init__(self, batch_size=5):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.batch_size = batch_size
        #self.policy=Network()
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
        self.states.append(tonumpy(state))
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(tonumpy(vals[0][0]))
        self.rewards.append(reward)
        self.done.append((tonumpy(done)))

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
        for _ in range(10):
            state,action,prob,values,rewards,entropy,batches=self.generate_batches()
            
            values=torch.from_numpy(values).to(device)
        
            for batch in batches:
                states=torch.from_numpy(state[batch]).to(device)
                actions=torch.from_numpy(action[batch]).to(device)
                probs=torch.from_numpy(prob[batch]).to(device)
                _,log_prob,critic_value,entropy=self.get_action_and_value(states,actions)
                old_probs=torch.exp(probs)
                print("OLD_PROBS")
                print(old_probs)
                new_probs=torch.exp(log_prob)
                print("new_probs")
                print(new_probs)
                r_t= new_probs/old_probs 
                print(r_t)           
                weighted_probs=advantage[batch]*r_t
                clip_weighted_probs=advantage[batch]*torch.clamp(r_t,0.8,1.2)
                actor_loss=-torch.min(weighted_probs,clip_weighted_probs).mean()

                returns=advantage[batch]+values[batch]
                critic_loss=(returns-critic_value.squeeze(0))**2
                critic_loss=critic_loss.mean()
                # ety=ety.mean()
                total_loss=actor_loss+0.5*critic_loss-0.05*entropy.mean()
                self.total_loss_wab=total_loss
                self.returns=returns.mean()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
            #print("epoch")
        self.clear_memory()

    def save_model(self, filename="ppo_checkpoint.pth"):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "log_std": self.log_std,
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




         
