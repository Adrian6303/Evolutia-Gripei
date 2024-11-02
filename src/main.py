import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Datele tale
input_path1 = 'Dataset\Dataset1.xlsx'
input_path2 = 'Dataset\Dataset2.xlsx'
output_path1 = 'output\output_Data1.txt'
output_path2 = 'output\output_Data2.txt'


def logistic_regression(inputpath, outputPath):
    # Citirea datelor
    data = pd.read_excel(inputpath, sheet_name='Sheet1')

    # Separați seturile de training și testing
    train_data = data[data['type'] == 'training']
    test_data = data[data['type'] == 'testing']

    # Selectarea caracteristicilor și a țintei
    X_train = train_data.drop(columns=['Donor ID', 'outcome', 'type'])
    y_train = train_data['outcome']
    X_test = test_data.drop(columns=['Donor ID', 'outcome', 'type'])
    y_test = test_data['outcome']


    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Antrenarea modelului Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predicții pe setul de testare
    predictions = model.predict(X_test_scaled)

    # Evaluare
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)


    # Checking feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)


    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Train data procentage:'+ str(X_train.shape[0] / data.shape[0]) + "\n")
        f.write('Test data procentage:'+ str(X_test.shape[0] / data.shape[0]) + "\n\n")
        
        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

        f.write("\n\nFeature importance:\n")
        f.write(feature_importance.to_string(index=False))


logistic_regression(input_path1, output_path1)
logistic_regression(input_path2, output_path2)