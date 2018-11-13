
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [16,10]


# In[3]:


rewards = list(np.load("penalty.npy"))


# In[4]:


plt.plot(rewards)


# In[6]:


sum(rewards)

