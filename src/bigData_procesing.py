import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE



def save_data(inputpath):
    df = pd.read_csv(inputpath)
    df = df.dropna(axis=1, how='all')  
    df['units'] = pd.to_numeric(df['units'], errors='coerce').fillna(0).astype(int)

    # Original pivot operation
    data_pivoted = df.pivot_table(index='donor_id', columns='name', values='data', aggfunc='last').reset_index()

    # Select only the unique 'donor_id' and 'units' columns from the original DataFrame
    units_column = df[['donor_id', 'units']].drop_duplicates(subset='donor_id')



    # Merge the 'units' column into the pivoted DataFrame
    data_pivoted = data_pivoted.merge(units_column, on='donor_id', how='left')
    vaccine_response = df[['donor_id', 'vaccine_response']].drop_duplicates()
    data = data_pivoted.merge(vaccine_response, on='donor_id', how='left')

    
    
    data = data[(data["vaccine_response"].isna() == False)]
    nr_drp = 0
    nr_fill = 0
    for column in data.columns:
        if data[column].isna().sum() >= len(data[column])*2/4:
            data.drop(column, axis=1, inplace=True)
            nr_drp += 1

        elif data[column].isna().sum() < len(data[column])*2 / 4 and data[column].dtype in ['float64', 'int64']:
            median_value = data[column].median()  # Calculate the median
            data[column].fillna(median_value, inplace=True)  # Fill the missing values with the median
            nr_fill += 1

    data.to_csv("Dataset\\Bazate\\data_procesata2.csv", index=False)   
    
    print(f"\n\n\nColumns dropped: {nr_drp}\nColumns filled: {nr_fill}") 


#test2 = drop when nan > 50% and fill with median when nan < 50%
#test3 = drop when nan > 75% and fill with median when nan < 75%

input_path = 'Dataset\\FluPRINT_database\\fluprint_export.csv'
save_data(input_path)