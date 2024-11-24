import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE


# Locatii fisiere
input_path1 = 'Dataset\\Small_Data\\Dataset1.xlsx'
input_path2 = 'Dataset\\Small_Data\\Dataset2.xlsx'
output_path1 = 'output\output_Data1.txt'
output_path2 = 'output\output_Data2.txt'
output_path4 = 'output\output_Data4.txt'
output_path5 = 'output\output_Data5.txt'
plot_path1 = 'output\learning_curve_Data1.png'
plot_path2 = 'output\learning_curve_Data2.png'
plot_path4 = 'output\learning_curve_Data4.png'
plot_path5 = 'output\learning_curve_Data5.png'

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
    print(np.unique(y_train), "\n\n\n\n")


    # Antrenarea modelului Logistic Regression
    model = LogisticRegression(max_iter=1000, penalty='l2', C=1.0, solver='liblinear', class_weight='balanced')

    #model = LogisticRegression(max_iter=1000)
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


def decision_tree(inputpath, outputPath, plotPath, max_depth, min_samples_split, min_samples_leaf):
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

    # Check class distribution
    print("Class distribution in training set:")
    print(y_train.value_counts())

    # Handle imbalanced dataset using SMOTE
    #smote = SMOTE()
    #X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train the model
    model = DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)

    # Plot learning curve and save as image
    plot_learning_curve(model, "Learning Curve (Decision Tree)", X_train, y_train, cv=StratifiedKFold(n_splits=5), output_path=plotPath)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluare
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    # Checking feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Train data procentage:'+ str(X_train.shape[0] / data.shape[0]) + "\n")
        f.write('Test data procentage:'+ str(X_test.shape[0] / data.shape[0]) + "\n\n")
        f.write("Max depth: " + str(max_depth) + "\n")
        f.write("Min samples split: " + str(min_samples_split) + "\n")
        f.write("Min samples leaf: " + str(min_samples_leaf) + "\n\n")
        
        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

        f.write("\n\nFeature importance:\n")
        f.write(feature_importance.to_string(index=False))


max_depth = 10
min_samples_split = 10
min_samples_leaf = 5

output_test1 = 'output\Small_Data\output_Data1-D-Test.txt'
output_test2 = 'output\Small_Data\output_Data2-D-Test.txt'
output_test3 = 'output\Small_Data\output_Data1-L-Test.txt'
output_test4 = 'output\Small_Data\output_Data2-L-Test.txt'
plot_test1 = 'output\Small_Data\learning_curve_Data1-D-Test.png'
plot_test2 = 'output\Small_Data\learning_curve_Data2-D-Test.png'
plot_path3 = 'output\Small_Data\learning_curve_Data1-L-Test.png'
plot_path4 = 'output\Small_Data\learning_curve_Data2-L-Test.png'


# Call the function
#decision_tree(input_path1, output_test1, plot_test1, max_depth, min_samples_split, min_samples_leaf)
#decision_tree(input_path2, output_test2, plot_test2, max_depth, min_samples_split, min_samples_leaf)

#logistic_regression(input_path1, output_test3, plot_path3)
#logistic_regression(input_path2, output_test4, plot_path4)