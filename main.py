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
    name="run-001",              
    config={
        "learning_rate": 0.0002,
        "batch_size": 200,
        "update_epochs": 10,
        "clip_coef": 0.2,
        "env": "Car Racing"
    }
)

i=0
k=0
#Trajectory created for 200 episodes each 
while k <episodes:
    while i<200:
        for j in range(1): #nuances in my code 
            state, reward, truncated, done= env.input()
            action,log_prob,value,entropy=memory.get_action_and_value(state.unsqueeze(0).to(device))
        
        state,reward,truncated,done=env.input(action)

        action,log_prob,value,entropy=memory.get_action_and_value(state.unsqueeze(0).to(device))
        # print(action,log_prob,value)
        done=truncated or done 
        memory.store_memory(state,action,log_prob,value,reward,done)
        i+=1
        if done:
            break
    memory.learn()
    i=0
    if done:
        env.dry_run()
        k=k+1
        wandb.log({
            "Total_loss": memory.total_loss_wab,
            "returns": memory.returns
            })

    
        

        






    
    

