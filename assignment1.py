import pandas as pd
from statsmodels.tsa.api import VAR
import plotly.express as px
import numpy as np


trainData = pd.read_csv("assignment_data_train.csv")
testData = pd.read_csv("assignment_data_test.csv")

trainData['Timestamp'] = pd.to_datetime(trainData['Timestamp'])
trainData.set_index(pd.DatetimeIndex(trainData['Timestamp']), inplace=True)

varData = trainData[['trips','month', 'day', 'hour']].dropna()[:-50]
model = VAR(varData)

lag = model.select_order()
print(lag.summary())

modelFit = model.fit(lag.aic)

nPeriods = varData.values[-modelFit.k_ar:]
pred = modelFit.forecast(nPeriods, steps = 50)





