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


def decision_tree(inputpath, outputPath, plotPath, max_depth, min_samples_split, min_samples_leaf, mean=False):
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
    if mean:
        threshold = 0.5 * len(data)
        valid_columns = data.columns[data.notna().sum() >= threshold]
       # data = data.dropna(axis=1, how='all')
        data = data.fillna(data.median())
        data[valid_columns] = data[valid_columns].apply(lambda col: col.fillna(col.median()), axis=0)

        #data = data.apply(lambda col: col.fillna(col.mean()), axis=0)

    data.to_csv('output\\Big_Data\\data2.csv', index=False)

    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    print(f"Număr de mostre în train_data după filtrare: {len(train_data)}\n\n\n\n")
    print(f"Număr de mostre în test_data după filtrare: {len(test_data)}\n\n\n\n")
    print(f"Număr de mostre în all_data după filtrare: {len(data)}\n\n\n\n")

    # Selectarea caracteristicilor și a țintei
    X_train = train_data.drop(columns=['donor_id', 'vaccine_response'])
    y_train = train_data['vaccine_response']
    X_test = test_data.drop(columns=['donor_id', 'vaccine_response'])
    y_test = test_data['vaccine_response']

    imputer = SimpleImputer(strategy="median")
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('imputer', imputer),
        ('dt', DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf))
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
        f.write("Max depth: " + str(max_depth) + "\n")
        f.write("Min samples split: " + str(min_samples_split) + "\n")
        f.write("Min samples leaf: " + str(min_samples_leaf) + "\n")
        if mean:
            f.write("Mean imputation: True\n")

        f.write("\nAcuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

        if mean:
            f.write("Mean imputation: True\n")

        f.write("\n\nFeature importance:\n")
        f.write(feature_importance.to_string(index=False))


def random_forest(inputpath, outputPath, plotPath, max_depth, min_samples_split, min_samples_leaf, mean=False):
    df = pd.read_csv(inputpath)
    df1 = df
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
    if mean:
        data = data.apply(lambda col: col.fillna(col.mean()), axis=0)

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
        ('dt', RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf))
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
        f.write("Max depth: " + str(max_depth) + "\n")
        f.write("Min samples split: " + str(min_samples_split) + "\n")
        f.write("Min samples leaf: " + str(min_samples_leaf) + "\n")
        if mean:
            f.write("Mean imputation: True\n")

        f.write("\nAcuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

        if mean:
            f.write("Mean imputation: True\n")

        f.write("\n\nFeature importance:\n")
        f.write(feature_importance.to_string(index=False))


outputTest = 'output\\Big_Data\\output_Data_D_Test.txt'
plotTest = 'output\\Big_Data\\learning_curve_D_Test.png'

outputTest2 = 'output\\Big_Data\\output_Data_F_Test.txt'
plotTest2 = 'output\\Big_Data\\learning_curve_F_Test.png'


max_depth = 100000000000
min_samples_split = 50
min_samples_leaf = 35

decision_tree(input_path, outputTest, plotTest, max_depth, min_samples_split, min_samples_leaf, mean=True)
#random_forest(input_path, outputTest2, plotTest2, max_depth, min_samples_split, min_samples_leaf, mean=True)

