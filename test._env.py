import gymnasium as gym
import numpy as np
import time
from environment_handler import Environment
env=Environment()
env.dry_run()
i=0
done= False 
while not done:
    obs,reward,truncated,done,info =env.input()
    done=truncated or done
    i+=1

print(info)
print(i)


# env=gym.make("CarRacing-v3", lap_complete_percent=1,render_mode ="human", domain_randomize=False, continuous=True, max_episode_steps=1050)
# obs,info=env.reset()

# arr=np.array([0,0,0])
# i=0
# done=False
# while not done:
#     if i<70:
#         obs,reward,truncated,done,info =env.step(arr)
#         # time.sleep(0.3)

#     else:
#         obs,reward,truncated,done,info =env.step(env.action_space.sample())
#     done= truncated or done

#     if i%200==0:
#         pass
#         # time.sleep(5)
#     i=i+1

# print(i)
