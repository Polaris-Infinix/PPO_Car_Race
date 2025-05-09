import torch 
import wandb
from environment_handler import Environment
from building_blocks import Network

# WandB Login 
wandb.login()
env=Environment()
env.dry_run()
act=Network()
done= False
while not done:
    for i in range(1): #nuances in my code 
        state, reward, truncated, done= env.input()
        action,log_prob,value=act.get_action_and_value(state.unsqueeze(0))
    # print(action)
    state,reward,truncated,done=env.input(action[0].detach().numpy())
    action,log_prob,value=act.get_action_and_value(state.unsqueeze(0))
    print(action,log_prob,value)



    done=truncated or done 
