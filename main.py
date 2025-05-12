import torch 
import wandb
from environment_handler import Environment
from building_blocks import *

# WandB Login 
wandb.login()
env=Environment()
env.dry_run()
act=Network()
memory=Memory()
done= False
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Trajectory created for 200 episodes each 
for i in range(200): 
    for i in range(1): #nuances in my code 
        state, reward, truncated, done= env.input()
        action,log_prob,value=act.get_action_and_value(state.unsqueeze(0).to(device))
    
    state,reward,truncated,entropy=env.input(action[0].detach().numpy())

    action,log_prob,value=act.get_action_and_value(state.unsqueeze(0).to(device))
    # print(action,log_prob,value)
    done=truncated or done 
    memory.store_memory(state,action,log_prob,value,reward,entropy)

adv=memory.advantages()
memory.learn()

    
    

