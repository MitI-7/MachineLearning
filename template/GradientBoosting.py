import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris, load_boston


def learning_gradient_boosting_classifier(X: np.ndarray, y: np.ndarray, cv=5) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier()
    return learning_gradient_boosting(model, X, y, cv)


def learning_gradient_boosting_regressor(X: np.ndarray, y: np.ndarray, cv=5) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor()
    return learning_gradient_boosting(model, X, y, cv)


def learning_gradient_boosting(model, X: np.ndarray, y: np.ndarray, cv=5):
    assert X.shape[0] == y.shape[0]
    assert cv > 0

    # 弱学習3000でハイパーパラメータのチューニング
    model.set_params(n_estimators=3000)
    parameters = {'learning_rate': [0.01, 0.02, 0.05, 0.1],
                  'max_depth': [4, 6],
                  'min_samples_leaf': [3, 5, 9, 17],
                  'max_features': [0.1, 0.3, 1.0]}

    gscv = GridSearchCV(model, parameters, verbose=1, n_jobs=-1, cv=cv)
    gscv.fit(X, y)
    print("hyper parameters")
    print("best score=", gscv.best_score_)
    print("best params=", gscv.best_params_)

    # 学習率のチューニング
    model = gscv.best_estimator_
    model.set_params(n_estimators=100000)
    parameters = {'learning_rate': [0.01, 0.02, 0.05, 0.1]}
    gscv = GridSearchCV(model, parameters, verbose=1, n_jobs=-1, cv=cv)
    gscv.fit(X, y)
    print("learning rate")
    print("best score=", gscv.best_score_)
    print("best params=", gscv.best_params_)

    return gscv.best_estimator_


def main():
    # 分類
    data = load_iris()
    X_train, X_test, y_train, X_test = train_test_split(data["data"], data["target"], train_size=0.1)
    model = learning_gradient_boosting_classifier(X_train, y_train)
    print(model.score(X_test, X_test))

    # 回帰
    data = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(data["data"], data["target"], train_size=0.1)
    model = learning_gradient_boosting_regressor(X_train, y_train)
    print(model.score(X_test, y_test))


if __name__ == '__main__':
    main()
