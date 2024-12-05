import pandas as pd
from joblib import load

model =load('output\\models\\logistic_regression2.joblib')
data = pd.read_csv('Dataset\\FluPRINT_database\\fluprint_export.csv')
#usec only the first 10 lines
data = data.head(10)
pred= model.predict(data)
#print the used columns by the model to predict
print(data.columns)
print(pred)