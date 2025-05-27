import torch 
import wandb
from building_blocks import *
import gymnasium as gym 

#hyperparameters
load=False
episode_length=250
episodes=2000
# WandB Login 
wandb.login()
memory=Memory()
done= False
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env=gym.make("LunarLander-v3",render_mode ="human")
wandb.init(
    project="Lunar_lander",          
    name="run-004",              
    config={
        "learning_rate": 0.0002,
        "batch_size": 200,
        "update_epochs": 10,
        "clip_coef": 0.2,
        "env": "Car Racing"
    }
)
total_rewards=[]
totalepisodeslength=[]
i,k=0,0
if load is True:
    memory.load_model() 

#Trajectory created for 200 episodes each 
while k < episodes:
    while i<episode_length:
        if i==0: #nuances in my code 
            state, info= env.reset()
            action,log_prob,value,entropy=memory.get_action_and_value(state)
        
        state,reward,truncated,done,info=env.step(action)
        action,log_prob,value,entropy=memory.get_action_and_value(state)
        done=truncated or done 
        memory.store_memory(state,action,log_prob,value,reward,entropy)
        i+=1
        if done:
            break
    total_rewards.append(memory.give_only_reward())
    totalepisodeslength.append(i)
    memory.learn()
    i=0
    
    if done:
        k=k+1
        total=sum(total_rewards)
        tol_epi_len=sum(totalepisodeslength)
        print(f"Episode:{k-1} Episode_length{tol_epi_len} Rewards gained {total}\n")
        wandb.log({
            "Episode_length":tol_epi_len,
            "Total_loss": memory.total_loss_wab,
            "returns": total,
            "Step": k-1
            })
        memory.save_model()
        total_rewards=[]
        totalepisodeslength=[]
