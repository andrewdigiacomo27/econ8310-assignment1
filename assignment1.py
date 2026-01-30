#GAM model - seems too detailed for this, did not complete

from pygam import LinearGAM, s, f, l
import patsy as pt

import pandas as pd
import numpy as np
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots


trainData = pd.read_csv("assignment_data_train.csv")
testData = pd.read_csv("assignment_data_test.csv")

# #generate x and y matrices
# eqn = """trips ~ -1 + year + month + day + hour"""
# y, x = pt.dmatrices(eqn, data = trainData)

#or can i do it this way
y = trainData['trips'].values
x = trainData[['year', 'month', 'day', 'hour']]

xTest = testData[['year', 'month', 'day', 'hour']]

#initializing and fit the model
model = LinearGAM(s(0) + s(1) + s(2) + s(3))
modelFit = model.gridsearch(np.asarray(x), y)

pred = modelFit.predict(xTest)


# rmse = np.sqrt(np.mean((pred - testData['trips'].values)**2))
# print("RMSE:", rmse)