
# coding: utf-8

# In[2]:


import opensim as osim
from osim.env import ProstheticsEnv
import gc
gc.enable()


# In[2]:


#import nbimporter
from ddpg import *


# In[3]:


EPISODES = 200
TEST = 10


# In[4]:


def main():
    env = ProstheticsEnv(visualize=False, difficulty=1)
    agent = DDPG(env)
#     env.monitor.start('experiments/' + ENV_NAME,force=True)
    
    # Playing Episodes
    for episode in range(EPISODES):
        state = env.reset()
        #print "episode:",episode
        # Train
        for step in range(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
                
        # Testing:
        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
    
    # Closing the monitor
#     env.monitor.close()


# In[5]:


if __name__ == '__main__':
    main()

