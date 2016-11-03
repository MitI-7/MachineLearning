import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV


def learning_logistic_regression(X: np.ndarray, y: np.ndarray, cv=5):
    parameters = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                  "penalty": ["l1", "l2"]}

    gscv = GridSearchCV(LogisticRegression(),
                        parameters,
                        cv=cv,
                        n_jobs=-1,
                        verbose=False)
    gscv.fit(X, y)
    print("LogisticRegression")
    print("best score=", gscv.best_score_)
    print("best params=", gscv.best_params_)

    return gscv.best_estimator_


def main():
    from sklearn.datasets import load_iris
    from sklearn.cross_validation import train_test_split

    data = load_iris()
    X, y = data["data"], data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = learning_logistic_regression(X_train, y_train)
    print(model.score(X_test, y_test))


if __name__ == '__main__':
    main()
