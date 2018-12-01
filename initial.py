import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def init_data(normalize=False):
    """
    Initializes the data.
    """
    data = pd.read_csv('train.csv')

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    if normalize:
        min_max_scaler = MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

    return X, y


def model_analysis(model, X, y):
    """
    Perform cross validation and score analysis for different algorithms.
    """
    model = model

    print("=" * 100, "\n", str(model), "\n", "=" * 100)

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    accuracies = []

    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = accuracy_score(y_test, y_pred)
        accuracies.append(score)

    accuracy = np.average(accuracies)
    print("Accuracy Score:", accuracy)


knn = KNeighborsClassifier()
lr = LogisticRegression()
d_tree = DecisionTreeClassifier()

X, y = init_data(normalize=False)

model_analysis(knn, X, y)
model_analysis(lr, X, y)
model_analysis(d_tree, X, y)
