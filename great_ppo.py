import gymnasium as gym 
import numpy as np 
import torch 
import wandb
import torch.nn as nn 
import torch.optim as optim 
from torch.distributions import Categorical


env=gym.make("LunarLander-v3")
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalize=True
learning_rate=1e-4
wandb.init(
    project="Lunar_lander",          
    name="run-88",              
    config={
        "learning_rate": learning_rate,
        "batch_size": 200,
        "update_epochs": 10,
        "clip_coef": 0.2,
        "env": "Car Racing",
        "Normalize":normalize
    }
)

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
optimizer=optim.Adam(model.parameters(),lr=learning_rate,eps=1e-5) 
cont=False
for m in range(500):
    #Tensors for storage
    n_steps=400
    states_t=torch.zeros(n_steps,8).to(device)
    action_t=torch.zeros(n_steps,1).to(device)
    reward_t=torch.zeros(n_steps,1).to(device)
    value_t=torch.zeros(n_steps,1).to(device)
    log_prob_t=torch.zeros(n_steps,1).to(device)
    done_t=torch.zeros(n_steps,1).to(device)
    t=0
    done=False
    while t< n_steps:
        done=False
        #main loop
        if cont==False:
            state, info=env.reset() 
        else:
            state=state_n

        while not done and t<n_steps:
            state=torch.tensor(state).unsqueeze(0).to(device)
            action, log_prob, _, value= model.get_actions_probs(state)
            state_n,reward, truncated, done, info =env.step(action.squeeze(0).cpu().numpy())
            done=truncated or done
            states_t[t]=state.squeeze(0)
            action_t[t]=action.detach()
            reward_t[t]=torch.tensor(reward)
            value_t[t] = value.detach()
            log_prob_t[t]=log_prob.detach()
            done_t[t]=done
            t=t+1   
            state=state_n
            if done:
                cont=False

    def adv():
        gamma=0.99
        gae_lambda=0.95
        next_done=done_t[-1]
        next_value=model.values(torch.tensor(state_n).unsqueeze(0).to(device)).detach()
        advantages = torch.zeros_like(reward_t)
        lastgaelam = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - done_t[t]
                nextvalues = value_t[t + 1]
            delta = reward_t[t] + gamma * nextvalues * nextnonterminal - value_t[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + value_t
        advantages=advantages.flatten()
        returns=returns.flatten()
        return advantages, returns

    reward_t=reward_t.flatten()
    log_prob_t=log_prob_t.flatten()
    action_t=action_t.flatten()
    advantages, returns=adv()
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


    wandb.log({
        "returns": returns.mean().item()
        })

    epoch=5
    cont=True if done_t[-1]==0  else False
    for _ in range(epoch):
        batch_size=5
        n_states = len(states_t)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + batch_size] for i in batch_start]
        
        for batch in batches:
            states_s= states_t[batch]
            log_prob_s=log_prob_t[batch]
            action_s=action_t[batch]
            actions,new_pros,entropy,new_value=model.get_actions_probs(states_s,action_s)
            r_t_log=new_pros-log_prob_s
            r_t=torch.exp(r_t_log)
            weighted_probs=-r_t*advantages[batch]
            weighted_probs_clip=-torch.clamp(r_t,0.8,1.2)*advantages[batch]
            actorloss=torch.max(weighted_probs,weighted_probs_clip).mean()
            critic_loss=(returns[batch]-new_value.flatten())**2
            critic_loss=critic_loss.mean()
            entropy_loss=entropy.mean()

            total_loss=actorloss+0.5*critic_loss-0.01*entropy_loss

            print(total_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            exit()


torch.save({
    "model_state_dict": model.state_dict(),
}, "ppo.pth")




# print(states_t.size())
# print(states_t)
# print(log_prob_t)
# print(action_t)
# print(reward_t)
# print(done_t)
# exit()
    

# print(nextvalue)
# print(states_t)
# print(state_n)
# print(states_t[-1])

# print(advantages.size())


# print(reward_t[3]*3)
# print(reward_t)


# print(returns)
# print(advantages)
# print(returns.size())
