
# coding: utf-8

# In[1]:


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


# In[35]:


def ob_dict_to_state(state_desc):

    res=[]

    res += [state_desc["target_vel"][0]- state_desc["body_vel"]['pelvis'][0]]
    res += [state_desc["target_vel"][2]- state_desc["body_vel"]['pelvis'][2]]

    res += [state_desc["target_vel"][0]]
    res += [state_desc["target_vel"][2]]

    pelvis_x_pos = state_desc["body_pos"]["pelvis"][0]
    pelvis_y_pos = state_desc["body_pos"]["pelvis"][1]
    pelvis_z_pos = state_desc["body_pos"]["pelvis"][2]

    for body_part in ["pelvis"]:
        res += state_desc["body_pos_rot"][body_part][:3] #ground_pelvis/pelvis_tilt/value in states file
        res += state_desc["body_vel_rot"][body_part][:3]
        res += state_desc["body_acc_rot"][body_part][:3]#2
        res += state_desc["body_acc"][body_part][0:3]

        #### for cyclical state, need to change pelvis_x_pos
        # res += [pelvis_x_pos]
        #####

        res += [state_desc["body_vel"][body_part][0]]
        res += [pelvis_y_pos]
        res += [state_desc["body_vel"][body_part][1]]
        res += [state_desc["body_vel"][body_part][2]]

    for body_part in ["head","torso", "pros_tibia_r","pros_foot_r","toes_l","talus_l"]:
        res += state_desc["body_pos_rot"][body_part][:3] #ground_pelvis/pelvis_tilt/value in states file
        res += state_desc["body_vel_rot"][body_part][:3]
        res += state_desc["body_acc_rot"][body_part][:3]#2
        res += state_desc["body_acc"][body_part][:3]
        res += [state_desc["body_pos"][body_part][0] - pelvis_x_pos]
        res += [state_desc["body_vel"][body_part][0]]
        res += [state_desc["body_pos"][body_part][1] - pelvis_y_pos]
        res += [state_desc["body_vel"][body_part][1]]
        res += [state_desc["body_pos"][body_part][2] - pelvis_z_pos]
        res += [state_desc["body_vel"][body_part][2]]

    #Only hip has more than one dof, but here last position is locked so not worth including 

    for joint in ["hip_r","knee_r","hip_l","knee_l","ankle_l"]: #removed back
        res += [state_desc["joint_pos"][joint][0] - pelvis_x_pos]
        #res += [state_desc["joint_pos"][joint][1] - pelvis_y_pos]
        #res += [state_desc["joint_pos"][joint][2] - pelvis_z_pos]
        res += state_desc["joint_pos"][joint][1:2]
        res += state_desc["joint_vel"][joint][:2]
        res += state_desc["joint_acc"][joint][:2] 

    mus_list = ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r', 'glut_max_r', 'iliopsoas_r', 'rect_fem_r', 'vasti_r', 'abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l', 'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l', 'gastroc_l', 'soleus_l', 'tib_ant_l']
    for muscle in mus_list:#state_desc["muscles"].keys():
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        # Add in muscle forces
        # res += state_desc['forces'][muscle]
    res += state_desc["forces"]["ankleSpring"]

    for foot in ['pros_foot_r_0','foot_l']:
        res += state_desc['forces'][foot][:6]

    cm_pos_x = [state_desc["misc"]["mass_center_pos"][0] - pelvis_x_pos]
    cm_pos_y = [state_desc["misc"]["mass_center_pos"][1] - pelvis_y_pos]
    cm_pos_z = [state_desc["misc"]["mass_center_pos"][2] - pelvis_z_pos]
    res = res + cm_pos_x + cm_pos_y + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

    head_behind_pen = min(0,state_desc["body_pos"]["head"][0] - pelvis_x_pos) * 0.5
#     str_leg_pen = 
    
    
    return res, head_behind_pen


# In[1]:


def main():
    env = ProstheticsEnv(visualize=True, difficulty=1)
    state_dim,_ = ob_dict_to_state(env.reset(project=False))
    state_dim = len(state_dim)
    agent = DDPG(env, state_dim)
    #env.monitor.start('experiments/' + ENV_NAME,force=True)
    
    # Playing Episodes
    try:
        train_rewards = list(np.load("train_rewards.npy"))
        avg_rewards = list(np.load("average_rewards.npy"))
        penalties = list(np.load("penalty.npy"))
    except:
        train_rewards = []
        avg_rewards = []
        penalties = []

    for episode in range(EPISODES):
        state = env.reset(project=False)
        state,_ = ob_dict_to_state(state)
        #print "episode:",episode
        # Train
        total_reward = 0
        total_penalty = 0
        for step in range(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action,project=False)
            next_state,penalty = ob_dict_to_state(next_state)
            agent.perceive(state,action,reward,next_state,done,episode)
            state = next_state
            total_reward += reward
            total_penalty += penalty
            if done:
                break
        train_rewards.append(total_reward)
        penalties.append(total_penalty)
                
        # Testing:
        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            # Running episodes
            for i in range(TEST):
                state = env.reset(project=False)
                state,_ = ob_dict_to_state(state)
                for j in range(env.spec.timestep_limit):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action,project=False)
                    state,_ = ob_dict_to_state(state)
                    total_reward += reward
                    if done:
                        break
            ave_reward = float(total_reward)/TEST
            avg_rewards.append(ave_reward)
            print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
        
        if episode % 100 == 0:
            print("Saving Rewards. Episode: ", episode)
            np.save("train_rewards.npy", train_rewards)
            np.save("average_rewards.npy", avg_rewards)
            np.save("penalty.npy", penalties)
            
        if episode % 100 == 0:
            print("Saving Memory. Episode: ", episode)
            agent.replay_buffer.save()
            
    # Closing the monitor
    #env.monitor.close()



# In[ ]:


if __name__ == '__main__':
    main()
    print("End Training")

