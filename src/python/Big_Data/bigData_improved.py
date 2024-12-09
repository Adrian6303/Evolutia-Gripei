from sklearn.calibration import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
# Import the plot_learning_curve function from the learning_curve.py file
from learning_curve import plot_learning_curve



def logistic_regression(inputpath, outputPath, plotPath, model_path):
    # Load data
    data = pd.read_csv(inputpath)  

    # Split data
    X = data.drop(columns=['vaccine_response', 'donor_id'])
    y = data['vaccine_response']


    # Define preprocessing for categorical and numerical columns
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Impute missing values and scale numeric features, OneHotEncode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Create a pipeline with preprocessing and logistic regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Fit the model
    model.fit(X_train, y_train)

    dump(model, model_path)

    # Plot learning curve
    plot_learning_curve(model, "Learning Curve (Logistic Regression)", X_train, y_train, cv=5, output_path=plotPath)

    # Predictions and evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

   


    # Save the classification report
    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Train data procentage:'+ str(X_train.shape[0] / data.shape[0]) + "\n")
        f.write('Test data procentage:'+ str(X_test.shape[0] / data.shape[0]) + "\n")
        f.write("Dataset: " + inputpath +  "\n\n")

        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

           
def decision_tree(inputpath, outputPath, plotPath, model_path, max_depth=100000000000, min_samples_split=35, min_samples_leaf=10):
    data = pd.read_csv(inputpath)

    categorical_columns = ['gender', 'race']

    # Convertim datele categorice folosind LabelEncoder
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)


    # Selectarea caracteristicilor și a țintei
    X_train = train_data.drop(columns=['donor_id', 'vaccine_response'])
    y_train = train_data['vaccine_response']
    X_test = test_data.drop(columns=['donor_id', 'vaccine_response'])
    y_test = test_data['vaccine_response']

    imputer = SimpleImputer(strategy="mean")
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('imputer', imputer),
        ('dt', DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf))
    ])
    model.fit(X_train, y_train)

    dump(model, model_path)

    # Plot learning curve
    plot_learning_curve(model, "Learning Curve (Decision Tree)", X_train, y_train, cv=5, output_path=plotPath)

    # Predicții pe setul de testare
    predictions = model.predict(X_test)

    # Evaluare
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)
    report = classification_report(y_test, predictions)

    # Checking feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.named_steps['dt'].feature_importances_
    }).sort_values(by='Importance', ascending=False)

    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Train data procentage:'+ str(X_train.shape[0] / data.shape[0]) + "\n")
        f.write('Test data procentage:'+ str(X_test.shape[0] / data.shape[0]) + "\n")
        f.write("Dataset: " + inputpath +  "\n\n")

        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("Max depth: " + str(max_depth) + "\n")
        f.write("Min samples split: " + str(min_samples_split) + "\n")
        f.write("Min samples leaf: " + str(min_samples_leaf) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

        f.write("\n\nFeature importance:\n")
        f.write(feature_importance.to_string(index=False))


def random_forest(inputpath, outputPath, plotPath, model_path, max_depth=100000000000, min_samples_split=50, min_samples_leaf=30):
    data = pd.read_csv(inputpath)
    categorical_columns = ['gender', 'race']

    # Convertim datele categorice folosind LabelEncoder
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)


    # Selectarea caracteristicilor și a țintei
    X_train = train_data.drop(columns=['donor_id', 'vaccine_response'])
    y_train = train_data['vaccine_response']
    X_test = test_data.drop(columns=['donor_id', 'vaccine_response'])
    y_test = test_data['vaccine_response']

    imputer = SimpleImputer(strategy="mean")
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('imputer', imputer),
        ('dt', RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf))
    ])
    model.fit(X_train, y_train)

    dump(model, model_path)

    # Plot learning curve
    plot_learning_curve(model, "Learning Curve (Random Forest)", X_train, y_train, cv=5, output_path=plotPath)

    # Predicții pe setul de testare
    predictions = model.predict(X_test)

    # Evaluare
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)
    report = classification_report(y_test, predictions)

    # Checking feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.named_steps['dt'].feature_importances_
    }).sort_values(by='Importance', ascending=False)

    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Train data procentage:'+ str(X_train.shape[0] / data.shape[0]) + "\n")
        f.write('Test data procentage:'+ str(X_test.shape[0] / data.shape[0]) + "\n")
        f.write("Dataset: " + inputpath +  "\n\n")

        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("Max depth: " + str(max_depth) + "\n")
        f.write("Min samples split: " + str(min_samples_split) + "\n")
        f.write("Min samples leaf: " + str(min_samples_leaf) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

        f.write("\n\nFeature importance:\n")
        f.write(feature_importance.to_string(index=False))



# File paths
input_path1 = 'Dataset\\Bazate\\NaN25.csv'
input_path2 = 'Dataset\\Bazate\\NaN50.csv'
input_path3 = 'Dataset\\Bazate\\NaN75.csv'
input_path4 = 'Dataset\\Bazate\\NaN25_DT_FI0.csv'
input_path5 = 'Dataset\\Bazate\\NaN50_DT_FI0.csv'
output_path = 'output\\Big_Data\\output_Data_L_improved.txt'
output_path2 = 'output\\Big_Data\\output_Data_DT_improved.txt'
output_path3 = 'output\\Big_Data\\output_Data_L2_improved.txt'
output_path4 = 'output\\Big_Data\\output_Data_DT2_improved.txt'
output_path5 = 'output\\Big_Data\\output_Data_L3_improved.txt'
output_path6 = 'output\\Big_Data\\output_Data_DT3_improved.txt'
output_path7 = 'output\\Big_Data\\output_Data_RF_improved.txt'
output_path8 = 'output\\Big_Data\\output_Data_RF2_improved.txt'
output_path9 = 'output\\Big_Data\\output_Data_RF3_improved.txt'
output_path10 = 'output\\Big_Data\\output_Data_DT_DrpFI0_improved.txt'
output_path11 = 'output\\Big_Data\\output_Data_DT2_DrpFI0_improved.txt'
plot_path = 'output\\Big_Data\\learning_curve_L_improved.png'
plot_path2 = 'output\\Big_Data\\learning_curve_DT_improved.png'
plot_path3 = 'output\\Big_Data\\learning_curve_L2_improved.png'
plot_path4 = 'output\\Big_Data\\learning_curve_DT2_improved.png'
plot_path5 = 'output\\Big_Data\\learning_curve_L3_improved.png'
plot_path6 = 'output\\Big_Data\\learning_curve_DT3_improved.png'
plot_path7 = 'output\\Big_Data\\learning_curve_RF_improved.png'
plot_path8 = 'output\\Big_Data\\learning_curve_RF2_improved.png'
plot_path9 = 'output\\Big_Data\\learning_curve_RF3_improved.png'
plot_path10 = 'output\\Big_Data\\learning_curve_DT_DrpFI0_improved.png'
plot_path11 = 'output\\Big_Data\\learning_curve_DT2_DrpFI0_improved.png'
model_path = 'output\\models\\lr_improved.joblib'
model_path2 = 'output\\models\\dt_improved.joblib'
model_path3 = 'output\\models\\lr2_improved.joblib'
model_path4 = 'output\\models\\dt2_improved.joblib'
model_path5 = 'output\\models\\lr3_improved.joblib'
model_path6 = 'output\\models\\dt3_improved.joblib'
model_path7 = 'output\\models\\rf_improved.joblib'
model_path8 = 'output\\models\\rf2_improved.joblib'
model_path9 = 'output\\models\\rf3_improved.joblib'
model_path10 = 'output\\models\\dt_drpFI0_improved.joblib'
model_path11 = 'output\\models\\dt2_drpFI0_improved.joblib'


#logistic_regression(input_path1, output_path, plot_path, model_path)
#decision_tree(input_path1, output_path2, plot_path2, model_path2)
#logistic_regression(input_path2, output_path3, plot_path3, model_path3)
#decision_tree(input_path2, output_path4, plot_path4, model_path4)
#logistic_regression(input_path3, output_path5, plot_path5, model_path5)
#decision_tree(input_path3, output_path6, plot_path6, model_path6)
#random_forest(input_path1, output_path7, plot_path7, model_path7)
#random_forest(input_path2, output_path8, plot_path8, model_path8)
#random_forest(input_path3, output_path9, plot_path9, model_path9)

#decision_tree(input_path4, output_path10, plot_path10, model_path10)
#decision_tree(input_path5, output_path11, plot_path11, model_path11)