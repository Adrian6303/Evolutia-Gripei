
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np


# Locatii fisiere
input_path1 = 'Dataset\\Small_Data\\Dataset1.xlsx'
input_path2 = 'Dataset\\Small_Data\\Dataset2.xlsx'
output_path = 'output\output_Data3.txt'
plot_path = 'output\learning_curve_Data3.png'

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
def logistic_regression2(inputpath1, inputpath2, outputPath, plotPath):
    # Citirea datelor
    train_data = pd.read_excel(inputpath1, sheet_name='Sheet1')
    test_data = pd.read_excel(inputpath2, sheet_name='Sheet1')

    # Selectarea caracteristicilor și a țintei
    X_train = train_data.drop(columns=['Donor ID', 'outcome', 'type',
        "CD57+ CD4+ T cells", "Th1 CXCR5+ CD8+ T cells", "CD16- CD56bright NK cells", "Th1 CXCR5- CD8+ T cells",
        "basophils", "CD16+ CD14- monocytes", "pDCs", "Th2 non- TFH CD4+ T cells", "CD57+ CD8+ T cells", "TFH CD4+ T cells",
        "PD1+ CD4+ T cells", "CD16+ CD14+ monocytes", "CD161+ NKT cells", "Th2 TFH CD4+ T cells", "ICOS+ CD8+ T cell",
        "mDCs", "Th17 CXCR5- CD8+ T cells", "ICOS+ CD4+ T cell", "Th17 TFH CD4+ T cells", "Th1 non- TFH CD4+ T cells",
        "CXCR5+ CD8+ T cells", "Th1 TFH CD4+ T cells", "Th17 non- TFH CD4+ T cells", "CD57+ NK cells", "Th2 CXCR5+ CD8+ T cells",
        "PD1+ CD8+ T cells", "Th2 CXCR5- CD8+ T cells", "Th17 CXCR5+ CD8+ T cells"])
    y_train = train_data['outcome']
    X_test = test_data.drop(columns=['Donor ID', 'outcome', 'type', 
        "IL6", "L50 ICAM1", "L50 TNFA", "L50 IL13", "L50 MCP1", "L50 IL10", 'L50 GROA', "L50 VEGF", 
        "L50 MCSF", "IL8", "L50 TGFA", "L50 IFNA", "L50 TNFB", "L50 IL5", "L50 IL1B", "L50 PDGFBB",
        "L50 IL4", "L50 IL6", "L50 RANTES", "L50 GMCSF", "L50 ENA78", "L50 LEPTIN", "L50 CD40L", "IL1B",
        "L50 IL7", "L50 IP10", "L50 MCP3", "L50 VCAM1", "L50 IFNB", "L50 NGF", "L50 IL1A", "L50 FGFB", "L50 IL1RA",
        "L50 FASL", "L50 IFNG", "L50 TGFB", "L50 MIG", "L50 LIF", "L50 IL17F", "L50 IL17", "L50 MIP1A", "L50 SCF", 
        "L50 RESISTIN", "L50 IL12P70", "L50 IL15", "L50 GCSF", "L50 IL2", "L50 IL8", "L50 IL12P40", "L50 EOTAXIN", 
        "L50 PAI1", "L50 MIP1B", "L50 HGF", "TNFA", "L50 TRAIL"])
    y_test = test_data['outcome']


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
        f.write('Train data procentage:'+ str(X_train.shape[0] / (X_train.shape[0] + X_test.shape[0])) + "\n")
        f.write('Test data procentage:'+ str(X_test.shape[0] / (X_train.shape[0] + X_test.shape[0])) + "\n\n")
        
        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")

        f.write("\n\nFeature importance:\n")
        f.write(feature_importance.to_string(index=False))


logistic_regression2(input_path1, input_path2, output_path, plot_path)