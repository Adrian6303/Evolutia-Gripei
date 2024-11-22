import pandas as pd
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



# Locatii fisiere
input_path = 'Dataset\\FluPRINT_database\\fluprint_export.csv'
output_path1 = 'output\\Big_Data\\output_Data_L.txt'
output_path2 = 'output\\Big_Data\\output_Data_D.txt'
plot_path1 = 'output\\Big_Data\\learning_curve_L.png'
plot_path2 = 'output\\Big_Data\\learning_curve_D.png'

# Functie pentru plotarea learning curve
def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5), output_path='learning_curve.png'):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if cv is None:
        cv = StratifiedKFold(n_splits=5)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(output_path)
    plt.close()


# Functie Logistic Regression

def logistic_regression_BIGDATA2(data, output, plot):
    
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    print(f"Număr de mostre în train_data după filtrare: {len(train_data)}\n\n\n\n")

    print(f"Număr de mostre în test_data după filtrare: {len(test_data)}\n\n\n\n")
    print(f"Număr de mostre în all_data după filtrare: {len(data)}\n\n\n\n")

    # Selectarea caracteristicilor și a țintei
    X_train = train_data.drop(columns=['donor_id', 'vaccine_response'])
    y_train = train_data['vaccine_response']
    X_test = test_data.drop(columns=['donor_id', 'vaccine_response'])
    y_test = test_data['vaccine_response']


    
    imputer = SimpleImputer(strategy="mean")
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('imputer', imputer),
        ('logreg', LogisticRegression(max_iter=2000))
    ])
    model.fit(X_train, y_train)

    # Plot learning curve
    plotPath = 'output\\BIGDATA\\' + plot
    plot_learning_curve(model, "Learning Curve (Logistic Regression)", X_train, y_train, cv=5, output_path=plotPath)


    # Predicții pe setul de testare
    predictions = model.predict(X_test)

    # Evaluare
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)
    report = classification_report(y_test, predictions)
        

    # Checking feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient':  model.named_steps['logreg'].coef_[0]
    }).sort_values(by='Coefficient', ascending=False)

    outputPath = 'output\\BIGDATA\\' + output
    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Train data procentage:'+ str(X_train.shape[0] / data.shape[0]) + "\n")
        f.write('Test data procentage:'+ str(X_test.shape[0] / data.shape[0]) + "\n\n")
        
        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

        f.write("\n\nFeature importance:\n")
        f.write(feature_importance.to_string(index=False))


def decision_tree_BIGDATA(inputpath, outputPath, plotPath):
    df = pd.read_csv(inputpath)
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

    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    print(f"Număr de mostre în train_data după filtrare: {len(train_data)}\n\n\n\n")
    print(f"Număr de mostre în test_data după filtrare: {len(test_data)}\n\n\n\n")
    print(f"Număr de mostre în all_data după filtrare: {len(data)}\n\n\n\n")

    # Selectarea caracteristicilor și a țintei
    X_train = train_data.drop(columns=['donor_id', 'vaccine_response'])
    y_train = train_data['vaccine_response']
    X_test = test_data.drop(columns=['donor_id', 'vaccine_response'])
    y_test = test_data['vaccine_response']

    imputer = SimpleImputer(strategy="mean")
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('imputer', imputer),
        ('dt', DecisionTreeClassifier())
    ])
    model.fit(X_train, y_train)

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
        f.write('Test data procentage:'+ str(X_test.shape[0] / data.shape[0]) + "\n\n")
        
        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

        f.write("\n\nFeature importance:\n")
        f.write(feature_importance.to_string(index=False))

def logistic_regression_BIGDATA(inputpath, outputPath, plotPath):
    df = pd.read_csv(inputpath)
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
    
    # Separați seturile de training și testing
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    print(f"Număr de mostre în train_data după filtrare: {len(train_data)}\n\n\n\n")
    print(f"Număr de mostre în test_data după filtrare: {len(test_data)}\n\n\n\n")
    print(f"Număr de mostre în total după filtrare: {len(data)}\n\n\n\n")

    # Selectarea caracteristicilor și a țintei
    X_train = train_data.drop(columns=['donor_id', 'vaccine_response'])
    y_train = train_data['vaccine_response']
    X_test = test_data.drop(columns=['donor_id', 'vaccine_response'])
    y_test = test_data['vaccine_response']


    
    imputer = SimpleImputer(strategy="mean")
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('imputer', imputer),
        ('logreg', LogisticRegression(max_iter=2000))
    ])
    model.fit(X_train, y_train)

    #Plot learning curve
    plot_learning_curve(model, "Learning Curve (Logistic Regression)", X_train, y_train, cv=5, output_path=plotPath)


    # Predicții pe setul de testare
    predictions = model.predict(X_test)

    # Evaluare
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)
    report = classification_report(y_test, predictions)
        

    # Checking feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient':  model.named_steps['logreg'].coef_[0]
    }).sort_values(by='Coefficient', ascending=False)


    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Train data procentage:'+ str(X_train.shape[0] / data.shape[0]) + "\n")
        f.write('Test data procentage:'+ str(X_test.shape[0] / data.shape[0]) + "\n\n")
        
        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

        f.write("\n\nFeature importance:\n")
        f.write(feature_importance.to_string(index=False))

#logistic_regression_BIGDATA(input_path, output_path1, plot_path1)
#decision_tree_BIGDATA(input_path, output_path2, plot_path2)