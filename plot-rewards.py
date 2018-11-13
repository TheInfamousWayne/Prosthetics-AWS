
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [16,10]


# In[2]:


rewards = list(np.load("train_rewards.npy"))


# In[3]:


plt.plot(rewards)


# In[6]:


sum(rewards)

