import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np


# Locatii fisiere
input_path4 = 'Dataset\\FluPRINT_database\\fluprint_export.csv'
output_path6 = 'output\\output_Data6.txt'
plot_path6 = 'output\\learning_curve_Data6.png'

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
def logistic_regression(inputpath, outputPath, plotPath):
    # Citirea datelor din csv
    data = pd.read_csv(inputpath)

    # Separați seturile de training și testing
    train_data = data[data['vaccine_response'] == '0' or data['vaccine_response'] == '1']
    test_data = data[data['vaccine_response'] == 'NULL']

    # Selectarea caracteristicilor și a țintei
    X_train = train_data.drop(columns=['donor_id', 'vaccine_response'])
    y_train = train_data['vaccine_response']
    X_test = test_data.drop(columns=['donor_id', 'vaccine_response'])
    y_test = test_data['vaccine_response']


    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Antrenarea modelului Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Plot learning curve
    plot_learning_curve(model, "Learning Curve (Logistic Regression)", X_train_scaled, y_train, cv=5, output_path=plotPath)


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


logistic_regression(input_path4, output_path6, plot_path6)