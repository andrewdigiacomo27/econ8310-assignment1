import pandas as pd
import numpy as np
from prophet import Prophet


trainData = pd.read_csv("assignment_data_train.csv")
testData = pd.read_csv("assignment_data_test.csv")

trainData['Timestamp'] = pd.to_datetime(trainData['Timestamp'])
trainData1 = trainData[['Timestamp', 'trips']]

trainData1 = pd.DataFrame(trainData1.values, columns = ['ds', 'y'])

model = Prophet(changepoint_prior_scale=0.5, daily_seasonality=True, weekly_seasonality=True)
model.fit(trainData1)

future = model.make_future_dataframe(periods= 744, freq= 'h')
pred = model.predict(future)

pred = pred['yhat'][-744:]
pred = np.array(pred)





