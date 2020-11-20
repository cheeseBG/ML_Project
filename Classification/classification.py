'''
Project Title : Bank Marketing
Author : Jang Jung Ik
Last Modified : 2020.11.
'''

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# Read Preprocessed dataset
df = pd.read_csv("../Data/After_bank.csv")

# Encoding categorical data to numerical data
encoding_df = df.apply(LabelEncoder().fit_transform)

new_df = encoding_df

# Split dataset
train_data, test_data = train_test_split(new_df, test_size=0.2)


def roc_curve_plot(model_name, fpr, tpr, roc_auc):
    plt.plot(fpr, tpr, linewidth=2, label='Area(AUC) = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.title(model_name + ' ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.show()


# ### RandomForest
def random_forest_cls(train_df, test_df):

    train_ft = train_df.drop(columns=['deposit'])
    train_tar = train_df['deposit']

    test_ft = test_df.drop(columns=['deposit'])
    test_tar = test_df['deposit']

    # K-fold validation (k = 10)
    kf = KFold(n_splits=10, shuffle=True)

    # Set parameter list
    param = {'criterion': ['gini', 'entropy'],
             'n_estimators': [1, 10, 20],
             'max_depth': [1, 2, 5, 10]}

    rf = RandomForestClassifier()

    # Set Grid Search
    grid_search = GridSearchCV(estimator=rf, param_grid=param,
                               cv=kf, n_jobs=4, verbose=2)

    # Fit the model
    grid_search.fit(train_ft, train_tar)

    print("Best parameters: " + str(grid_search.best_params_))
    print("Best score: " + str(grid_search.best_score_))

    # Predict with test data
    pred = grid_search.best_estimator_.predict(test_ft)

    # Get ROC acurracy score
    probs = grid_search.best_estimator_.predict_proba(test_ft)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(test_tar, preds)
    roc_auc = auc(fpr, tpr)

    # Show ROC curve plot
    roc_curve_plot('Random Forest', fpr, tpr, roc_auc)

    # Display confusion matrix
    print("\n\n< Random Forest Confusion matrix >")
    print(confusion_matrix(test_tar, pred))
    print(classification_report(test_tar, pred))

    return grid_search.best_estimator_, roc_auc, fpr, tpr


# ### Logistic Regression
def logistic_regression_cls(train_df, test_df):

    train_ft = train_df.drop(columns=['deposit'])
    train_tar = train_df['deposit']

    test_ft = test_df.drop(columns=['deposit'])
    test_tar = test_df['deposit']

    # K-fold validation (k = 10)
    kf = KFold(n_splits=10, shuffle=True)

    # Set parameter list
    param = {'C': [0.1, 1.0, 10.0],
             'solver': ['liblinear', 'lbfgs', 'sag'],
             'max_iter': [50, 100, 200]}

    lr = LogisticRegression()

    # Set Grid Search
    grid_search = GridSearchCV(estimator=lr, param_grid=param,
                               cv=kf, n_jobs=4, verbose=2)

    # Fit the model
    grid_search.fit(train_ft, train_tar)

    print("Best parameters: " + str(grid_search.best_params_))
    print("Best score: " + str(grid_search.best_score_))

    # Predict with test data
    pred = grid_search.best_estimator_.predict(test_ft)

    # Get ROC acurracy score
    probs = grid_search.best_estimator_.predict_proba(test_ft)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(test_tar, preds)
    roc_auc = auc(fpr, tpr)

    # Show ROC curve plot
    roc_curve_plot('Logistic Regression', fpr, tpr, roc_auc)

    # Display confusion matrix
    print("\n\n< Logistic Regression Confusion matrix >")
    print(confusion_matrix(test_tar, pred))
    print(classification_report(test_tar, pred))

    return grid_search.best_estimator_, roc_auc, fpr, tpr


# ### SVM
def svm_cls(train_df, test_df):

    train_ft = train_df.drop(columns=['deposit'])
    train_tar = train_df['deposit']

    test_ft = test_df.drop(columns=['deposit'])
    test_tar = test_df['deposit']

    # K-fold validation (k = 10)
    kf = KFold(n_splits=10, shuffle=True)

    # Set parameter list
    # param = {'C': [0.1, 1.0],
    #          'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #          'gamma': [0.01, 0.1, 1.0]}
    param = {'C': [0.1],
             'kernel': ['sigmoid'],
             'gamma': [0.01, 0.1]}

    svm = SVC(probability=True)

    # Set Grid Search
    grid_search = GridSearchCV(estimator=svm, param_grid=param,
                               cv=kf, n_jobs=4, verbose=2)

    # Fit the model
    grid_search.fit(train_ft, train_tar)

    print("Best parameters: " + str(grid_search.best_params_))
    print("Best score: " + str(grid_search.best_score_))

    # Predict with test data
    pred = grid_search.best_estimator_.predict(test_ft)

    # Get ROC acurracy score
    probs = grid_search.best_estimator_.predict_proba(test_ft)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(test_tar, preds)
    roc_auc = auc(fpr, tpr)

    # Show ROC curve plot
    roc_curve_plot('SVM', fpr, tpr, roc_auc)

    # Display confusion matrix
    print("\n\n< SVM Confusion matrix >")
    print(confusion_matrix(test_tar, pred))
    print(classification_report(test_tar, pred))

    return grid_search.best_estimator_, roc_auc, fpr, tpr


def ensemble_cls(train_df, test_df, best_rf, best_lr, best_svm):
    train_ft = train_df.drop(columns=['deposit'])
    train_tar = train_df['deposit']

    test_ft = test_df.drop(columns=['deposit'])
    test_tar = test_df['deposit']

    # Define voting classifier
    voting_cls = VotingClassifier(estimators=[('sv', best_svm), ('lr', best_lr), ('rf', best_rf)], voting='soft')

    # Fit the model
    voting_cls.fit(train_ft, train_tar)

    # Predict with test data
    voting_result = voting_cls.predict(test_ft)

    # Get ROC acurracy score
    probs = voting_cls.predict_proba(test_ft)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(test_tar, preds)
    roc_auc = auc(fpr, tpr)

    # Show ROC curve plot
    roc_curve_plot('Ensemble', fpr, tpr, roc_auc)

    # Display confusion matrix
    print("\n\n< Confusion matrix of ensemble result >")
    print(confusion_matrix(test_tar, voting_result))
    print(classification_report(test_tar, voting_result))

    return roc_auc, fpr, tpr


rf_best_estimator, rf_score, rf_fpr, rf_tpr = random_forest_cls(train_data, test_data)
lr_best_estimator, lr_score, lr_fpr, lr_tpr = logistic_regression_cls(train_data, test_data)
svm_best_estimator, svm_score, svm_fpr, svm_tpr = svm_cls(train_data, test_data)
ensemble_score, ensemble_fpr, ensemble_tpr = ensemble_cls(train_data, test_data, rf_best_estimator, lr_best_estimator, svm_best_estimator)

# Comparison ROC Curve
plt.plot(rf_fpr, rf_tpr, linewidth=2, label='RF, Area(AUC) = %0.2f' % rf_score, color='red')
plt.plot(lr_fpr, lr_tpr, linewidth=2, label='LR, Area(AUC) = %0.2f' % lr_score, color='green')
plt.plot(svm_fpr, svm_tpr, linewidth=2, label='SVM, Area(AUC) = %0.2f' % svm_score, color='blue')
plt.plot(ensemble_fpr, ensemble_tpr, linewidth=2, label='Area(AUC) = %0.2f' % ensemble_score, color='yellow')
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.title('Comparison ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

plt.show()

# Print Model Ranking(by ROC Accuracy)
score_list = [rf_score, lr_score, svm_score, ensemble_score]
score_list.sort(reverse=True)
rank = 1

print("< Model Ranking >")
for s in score_list:
    if s == rf_score:
        print(str(rank) + '. Random Forest ' + str(s))
    elif s == lr_score:
        print(str(rank) + '. Logistic Regression ' + str(s))
    elif s == svm_score:
        print(str(rank) + '. SVM ' + str(s))
    elif s == ensemble_score:
        print(str(rank) + '. Ensemble ' + str(s))
    rank += 1