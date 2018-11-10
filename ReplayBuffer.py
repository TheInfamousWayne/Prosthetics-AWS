
# coding: utf-8

# In[1]:


from collections import deque
import random
import pickle as pickle


# In[2]:


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
        self.load()
        
    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return self.buffer_size
    
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            
    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences
    
    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0 
        
    def save(self):
        pickle.dump([self.buffer,self.num_experiences], open("./replay_memory.pickle", 'wb'), True)
        print('memory dumped into ', "./replay_memory.pickle")
    
    def load(self):
        try:
            [self.buffer,self.num_experiences] = pickle.load(open("./replay_memory.pickle", 'rb'))
            print('memory loaded from ',"./replay_memory.pickle")
        except:
            print("a new beginning")
