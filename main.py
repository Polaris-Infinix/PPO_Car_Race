import torch 
import wandb
from building_blocks import *
import gymnasium as gym 

#hyperparameters
load=False
episode_length=2000
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
    state,info=env.reset()
    while not done:
        action, log_prob, value,_=memory.get_action_and_value(state)
        state_n,reward, truncated,done, info=env.step(action)
        done =done  or truncated
        memory.store_memory(state, action, log_prob, value, reward, done)
        state=state_n
        i+=1
        
    total_rewards.append(memory.give_only_reward())
    totalepisodeslength.append(i)
    memory.learn()
    i=0
    k+=1
    total=sum(total_rewards)
    tol_epi_len=sum(totalepisodeslength)
    print(f"Episode:{k} Episode_length{tol_epi_len} Rewards gained {total}\n")
    wandb.log({
        "Episode_length":tol_epi_len,
        "Actor Loss": memory.actor_losses,
        "Critic Loss":memory.critic_losses,
        "Entropy":memory.entropies,
        "returns": total,
        })
    memory.save_model()
    total_rewards=[]
    totalepisodeslength=[]
    done=False
