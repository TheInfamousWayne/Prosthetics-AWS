
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [16,10]


# In[4]:


rewards = list(np.load("penalty.npy"))


# In[5]:


plt.plot(rewards)


# In[35]:


import tensorflow as tf


# In[36]:


from osim.env import ProstheticsEnv


# In[37]:


env = ProstheticsEnv(visualize=False, difficulty=1)


# In[38]:


env.spec.timestep_limit


# In[ ]:




