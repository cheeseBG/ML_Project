'''
Project Title : Bank Marketing
Author : Jang Jung Ik
Last Modified : 2020.11.
'''

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("bank-full.csv")

feature = df.drop(columns=['y'])
tar = df['y'].values

encoding_ft = feature.apply(LabelEncoder().fit_transform)

scaler = StandardScaler()
scaled_ft = scaler.fit_transform(encoding_ft)
scaled_ft = pd.DataFrame(scaled_ft, columns=encoding_ft.columns, index=list(encoding_ft.index.values))

new_df = scaled_ft

new_df['y'] = tar

new_df = new_df.drop(columns=['education', 'default', 'housing',
                                'loan', 'contact', 'day', 'month',
                                'campaign', 'pdays', 'previous',
                                'poutcome'])

train_df, test_df = train_test_split(new_df, test_size=0.2)


# RandomForest
def random_forest_cls(train_df, test_df):

    train_ft = train_df.drop(columns=['y'])
    train_tar = train_df['y'].values

    test_ft = test_df.drop(columns=['y'])
    test_tar = test_df['y'].values

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

    pred = grid_search.best_estimator_.predict(test_ft)

    # Display confusion matrix
    print("\n\n< Confusion matrix >")
    print(confusion_matrix(test_tar, pred))
    print(classification_report(test_tar, pred))

    return grid_search.best_estimator_


def logistic_regression_cls(df):

    df_ft = df.drop(columns=['y'])
    df_tar = df['y'].values

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
    grid_search.fit(df_ft, df_tar)

    print("Best parameters: " + str(grid_search.best_params_))
    print("Best score: " + str(grid_search.best_score_))

    return grid_search.best_estimator_


def svm_cls(df):

    df_ft = df.drop(columns=['y'])
    df_tar = df['y'].values

    # K-fold validation (k = 10)
    kf = KFold(n_splits=10, shuffle=True)

    # Set parameter list
    param = {'C': [0.1, 1.0, 10.0],
             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
             'gamma': [0.01, 0.1, 1.0, 10.0]}

    svm = SVC()

    # Set Grid Search
    grid_search = GridSearchCV(estimator=svm, param_grid=param,
                               cv=kf, n_jobs=4, verbose=2)

    # Fit the model
    grid_search.fit(df_ft, df_tar)

    print("Best parameters: " + str(grid_search.best_params_))
    print("Best score: " + str(grid_search.best_score_))

    return grid_search.best_estimator_


def ensemble_cls(train_df, test_df, best_rf, best_lr, best_svm):
    train_ft = train_df.drop(columns=['y'])
    train_tar = train_df['y'].values

    test_ft = test_df.drop(columns=['y'])
    test_tar = test_df['y'].values

    voting_cls = VotingClassifier(estimators=[('sv', best_svm), ('lr', best_lr), ('rf', best_rf)], voting='hard')

    voting_cls.fit(train_ft, train_tar)

    voting_result = voting_cls.predict(test_ft)

    # Display confusion matrix
    print("\n\n< Confusion matrix of ensemble result >")
    print(confusion_matrix(test_tar, voting_result))
    print(classification_report(test_tar, voting_result))
