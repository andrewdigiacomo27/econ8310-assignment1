import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
import plotly.express as px
import numpy as np

#this should work. need to find a way to adjust accuracy

trainData = pd.read_csv("assignment_data_train.csv")
testData = pd.read_csv("assignment_data_test.csv")

trainData['Timestamp'] = pd.to_datetime(trainData['Timestamp'])
trainTrips = trainData['trips']

#maybe remove seasonality is causing error? test RMSE
model = ExponentialSmoothing(trainTrips, trend='add')

modelFit = model.fit(optimized = True)

forecast1 = len(testData)
pred = modelFit.forecast(forecast1)
pred = np.array(pred)