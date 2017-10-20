import numpy as np


def sigmoid(x):
    x = max(min(x, 35), -35)
    return 1.0 / (1.0 + np.exp(-x))


class FTRLProximal:
    """Per-Coordinate FTRL_proximal with L1 and L2 Regularization for Logistic Regression
       https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
       https://www.kaggle.com/c/avazu-ctr-prediction/discussion/10927
       https://github.com/jeongyoonlee/Kaggler/blob/master/kaggler/online_model/ftrl.pyx
    """
    def __init__(self, alpha: float=0.1, beta: float=1.0, L1: float=1.0, L2: float=1.0, D: int=2**20):
        assert alpha != 0.0

        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        self.n = np.zeros((D,), dtype=np.float64)
        self.z = np.zeros((D,), dtype=np.float64)
        self.w = np.zeros((D,), dtype=np.float64)

    def __repr__(self):
        return f"FTRLProximal(a={self.alpha}, b={self.beta}, l1={self.L1}, l2={self.L2})"

    def predict(self, x: dict) -> float:
        x[0] = 1.0  # bias
        wTx = sum(self.w[k] * v for k, v in x.items())
        return sigmoid(wTx)

    def update(self, x: dict, p: float, y: float):
        x[0] = 1.0  # bias
        for i, v in x.items():
            g = (p - y) * v

            # update z & n
            sigma = (np.sqrt(self.n[i] + g * g) - np.sqrt(self.n[i])) / self.alpha
            self.z[i] += g - sigma * self.w[i]
            self.n[i] += g * g

            # update w
            sign = -1.0 if self.z[i] < 0 else 1.0
            if sign * self.z[i] <= self.L1:
                self.w[i] = 0.0
            else:
                self.w[i] = (sign * self.L1 - self.z[i]) / ((self.beta + np.sqrt(self.n[i])) / self.alpha + self.L2)

    def fit(self, X, y, epoch=1):
        for e in range(epoch):
            for _y, x in zip(y, X):
                p = self.predict(x)
                self.update(x, p, _y)


def main():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    X_train_dict = [{i: x[i] for i in range(len(x))} for x in X_train]
    X_test_dict = [{i: x[i] for i in range(len(x))} for x in X_test]

    model = FTRLProximal()
    model.fit(X=X_train_dict, y=y_train, epoch=50)

    y_pred = []
    for x in X_test_dict:
        p = model.predict(x)
        y_pred.append(p)

    print(confusion_matrix(y_test, [1 if p > 0.5 else 0 for p in y_pred]))


if __name__ == '__main__':
    main()
