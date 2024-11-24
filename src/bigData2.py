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
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - np.std(train_scores, axis=1),
                     train_scores_mean + np.std(train_scores, axis=1), alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - np.std(test_scores, axis=1),
                     test_scores_mean + np.std(test_scores, axis=1), alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig(output_path)
    plt.close()

def logistic_regression2(inputpath, outputPath, plotPath):
    # Load data
    data = pd.read_csv(inputpath)  
    data = data.dropna(subset=['vaccine_response'])  # Drop rows without target

    # Split data
    X = data.drop(columns=['vaccine_response', 'donor_id', 'study_id', 'mesurment_id', 'visit_id'])
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
        f.write("Model Accuracy: " + str(accuracy) + "\n")
        f.write("\nClassification Report:\n" + report)



input_path = 'Dataset\\FluPRINT_database\\fluprint_export.csv'
output_path1 = 'output\\Big_Data\\output_Data_L-2.txt'
plot_path1 = 'output\\Big_Data\\learning_curve_L-2.png'
model_path = 'output\\models\\logistic_regression2.joblib'


logistic_regression2(input_path, output_path1, plot_path1)