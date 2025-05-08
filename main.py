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
    state, reward, truncated, done= env.input()
    act.get_action_and_value(state)

    done=truncated or done 
