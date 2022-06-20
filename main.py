# Source: https://www.kaggle.com/code/kerneler/starter-student-life-5734244e-2

# Questions
# What are the most important attributes for an Ideal Student Life ?
# What factors leads to Stress, More Participation, More Interaction,
# More Satisfaction ?
# Does more stress leads to less satisfaction?
# What factors lead to increased stress among the students?

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from utils import *

for dirname, _, filenames in os.walk('D:\StudentLifeDataset\dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


nRowsRead = 1000  # specify 'None' if want to read whole file
# conversation_u00.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('D:\StudentLifeDataset\dataset\sensing\conversation\conversation_u00.csv',
                  delimiter=',', nrows=nRowsRead)
df1.dataframeName = 'conversation_u00.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

# Let's take a quick look at what the data looks like:
df1.head(5)

# Distribution graphs (histogram/bar graph) of sampled columns:
plotPerColumnDistribution(df1, 10, 5)

# Correlation matrix:
plotCorrelationMatrix(df1, 8)

# Scatter and density plots:
plotScatterMatrix(df1, 6, 15)
