
# coding: utf-8

# In[2]:

# read data into a DataFrame
import pandas as pd
import pylab as plt
import seaborn
from sklearn.linear_model import LinearRegression
import numpy.random as nprnd
import random
import json
import numpy as np
pd.set_option('display.max_columns', 500)
get_ipython().magic(u'matplotlib inline')

df = pd.read_csv('https://github.com/Columbia-Intro-Data-Science/python-introduction-mailman82/blob/master/Project/CHSHousingProjectData.csv')
df.head()


# In[ ]:



