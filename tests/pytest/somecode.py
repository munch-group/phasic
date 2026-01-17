#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cppimport.import_hook
import somecode #This will pause for a moment to compile the module
somecode.square(9)


# In[ ]:





# In[ ]:





# In[5]:


import phasic
import cppimport.import_hook
import rabbit_state_space #This will pause for a moment to compile the module
graph = phasic.Graph(rabbit_state_space.build(2, 2, 4))


# In[7]:


graph.plot()


# In[ ]:




