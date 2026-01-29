import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX

# Load data
trainData = pd.read_csv("assignment_data_train.csv")
testData = pd.read_csv("assignment_data_test.csv")

trainData['Timestamp'] = pd.to_datetime(trainData['Timestamp'])
testData['Timestamp'] = pd.to_datetime(testData['Timestamp'])

# Endogenous variables (VAR part)
endog = trainData[['trips', 'hour']]

# Exogenous variables (X part)
exog = trainData[['month', 'day']]

# --------------------
# Define model
# --------------------
model = VARMAX(
    endog,
    exog=exog,
    order=(1, 1),      # VARMA(1,1)
    trend='c'
)

# Fit model
modelFit = model.fit(disp=False)

# --------------------
# Forecast January
# --------------------
exog_future = testData[['month', 'day']]

forecast = modelFit.forecast(
    steps=744,
    exog=exog_future
)

# Extract trips forecast only
pred = forecast['trips'].values