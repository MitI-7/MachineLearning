from datetime import datetime
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_iris
from sklearn.externals import joblib
from scipy.sparse import hstack
from LogisticRegression import learning_logistic_regression
from GradientBoosting import learning_gradient_boosting_classifier


class FeatureTransformerUsingGBDT:
    def __init__(self, learned_model):
        assert len(learned_model.estimators_) > 0, "modelが学習されていない"

        self.model = learned_model
        self.label_binarizer_list = None

    # 各弱学習器の出力をone-hotにするLabelBinarizerを作成
    def fit(self, X: np.ndarray) -> None:
        self.label_binarizer_list = []
        estimators = np.asarray(self.model.estimators_).ravel()
        for estimator in estimators:
            leaf = estimator.tree_.apply(X)
            lb = LabelBinarizer(sparse_output=True)
            lb.fit_transform(leaf)
            self.label_binarizer_list.append(lb)

    # 素性変換
    def transform(self, X: np.ndarray) -> np.ndarray:
        feature_list = []
        estimators = np.asarray(self.model.estimators_).ravel()
        for estimator, lb in zip(estimators, self.label_binarizer_list):
            feature_list.append(lb.transform(estimator.tree_.apply(X)))
        return hstack(feature_list).toarray()


def main():
    data = load_iris()
    X = data["data"].astype(np.float32)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print("素性変換器の作成")
    #model = learning_gradient_boosting_classifier(X, y)
    model = GradientBoostingClassifier()
    model.fit(X, y)
    feature_transformer = FeatureTransformerUsingGBDT(model)
    feature_transformer.fit(X_train)

    print("素性の変換")
    X_train_tf = feature_transformer.transform(X_train)
    X_test_tf = feature_transformer.transform(X_test)

    print("ロジスティック回帰の学習")
    model = learning_logistic_regression(X_train_tf, y_train)
    print("score:", model.score(X_test_tf, y_test))

    model = learning_logistic_regression(X_train, y_train)
    print("score:", model.score(X_test, y_test))

if __name__ == '__main__':
    main()
