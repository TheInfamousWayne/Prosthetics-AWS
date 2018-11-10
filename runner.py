
# coding: utf-8

# In[2]:


import opensim as osim
from osim.env import ProstheticsEnv
import pandas as pd
import numpy as np
import gc
gc.enable()


# In[2]:


#import nbimporter
from ddpg import *


# In[3]:


EPISODES = 10001
TEST = 100


# In[4]:


def main():
    env = ProstheticsEnv(visualize=False, difficulty=1)
    agent = DDPG(env)
    #env.monitor.start('experiments/' + ENV_NAME,force=True)
    
    # Playing Episodes
    try:
        train_rewards = list(np.load("train_rewards.npy"))
        avg_rewards = list(np.load("average_rewards.npy"))
    except:
        train_rewards = []
        avg_rewards = []

    for episode in range(EPISODES):
        state = env.reset()
        #print "episode:",episode
        # Train
        total_reward = 0
        for step in range(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            total_reward += reward
            if done:
                break
        train_rewards.append(total_reward)
        
                
        # Testing:
        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            # Running episodes
            for i in range(TEST):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = float(total_reward)/TEST
            avg_rewards.append(ave_reward)
            print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
        
        if (episode+1) % 200 == 0:
            np.save("train_rewards.npy", train_rewards)
            np.save("average_rewards.npy", avg_rewards)
            
        if (episode+1) % 2000 == 0:
            agent.replay_buffer.save()
            
    # Closing the monitor
    #env.monitor.close()


# In[5]:


if __name__ == '__main__':
    main()

