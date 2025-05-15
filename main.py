import torch 
import wandb
from environment_handler import Environment
from building_blocks import *

# WandB Login 
wandb.login()
env=Environment()
env.dry_run()
memory=Memory()
done= False
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
episode_length=200
episodes=2000

wandb.init(
    project="ppo-atari",          
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
memory.load_model()
#Trajectory created for 200 episodes each 
while k < episodes:
    while i<200:
        for j in range(1): #nuances in my code 
            state, reward, truncated, done= env.input()
            action,log_prob,value,entropy=memory.get_action_and_value(state.unsqueeze(0).to(device))
        
        state,reward,truncated,done=env.input(action)

        action,log_prob,value,entropy=memory.get_action_and_value(state.unsqueeze(0).to(device))
        done=truncated or done 
        memory.store_memory(state,action,log_prob,value,reward,entropy)
        i+=1
        if done:
            break
    total_rewards.append(memory.give_only_reward())
    totalepisodeslength.append(i)
    print(totalepisodeslength,total_rewards)
    memory.learn()
    i=0
    
    if done:
        env.dry_run()
        k=k+1
        total=sum(total_rewards)
        tol_epi_len=sum(totalepisodeslength)
        wandb.log({
            "Episode_length":tol_epi_len,
            "Total_loss": memory.total_loss_wab,
            "returns": total,
            "Step": k-1
            })
        memory.save_model()
        total_rewards=[]
        totalepisodeslength=[]
