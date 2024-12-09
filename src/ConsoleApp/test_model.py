import os
import pandas as pd
from joblib import load

model =load('src\\output\\models\\lr3_improved.joblib')
data = pd.read_csv('src\\Dataset\\ProcessedDatasets\\NaN75.csv')

# #usec only the 40 th line
# data = data.iloc[38:39]
# data.drop(columns=['donor_id', 'vaccine_response', 'visit_age', 'gender', 'race'], inplace=True)
# #print(data)
# #save data in file
# data.to_csv('src\\ConsoleApp\\2pacient.csv', index=False)


visit_age = input("Enter the pacient age: ")
while not visit_age.isdigit() or int(visit_age) < 0 or int(visit_age) > 150:
    visit_age = input("Enter a valid age: ")

gender = input("Enter the pacient gender: ")
while gender not in ['Female', 'Male']:
    gender = input("Enter a valid gender (Male or Female): ")
race = input("Enter the pacient race: ")
while race not in ["Caucasian", "Asian", "American Indian or Alaska Native", "Black or African American", "Hispanic/Latino", "Other"]:
    race = input("Enter a valid race: ")
data_path = input("Enter the path to the file with the pacient data: ")
while not data_path.endswith('.csv') or not os.path.exists(data_path):
    data_path = input("Enter a valid path to a csv file: ")
data = pd.read_csv(data_path)
#add to data the fields visit_age , gender , race
data["visit_age"] = visit_age
data["gender"] = gender
data["race"] = race
# print(data)
pred= model.predict(data)
#print the used columns by the model to predict
#print(data.columns)
# print(pred)
print("\n For the pacient with age: ", visit_age , ", gender: ", gender ,", and race: ", race , " the model predicted: ")
if(pred[0]==0):
    print("The patient is not having a problem with the vaccine")
else:
    print("The patient is having a problem with the vaccine")