import numpy as np
import random

from numpy.random import beta


np.set_printoptions(precision=3, suppress=True, linewidth=100)


class LogisticRegression:
    def __init__(
        self,
        penalty=None,
        learning_rate=0.001,
        n_itr=2000,
        batch_size=None,
        tol=1e-6,
        C=1.0,
    ):
        self.lr = self._validate_positive(learning_rate, "learning_rate")
        self.n_itr = int(self._validate_positive(n_itr, "n_itr"))
        self.batch_size = self._validate_positive(batch_size, "batch_size")
        self.penalty = self._validate_penalty(penalty)
        self.tol = self._validate_positive(tol, "tol")
        self.C = self._validate_positive(C, "C")
        self.beta = None
        self.intercept = 0
        self.losses = []

    def _validate_positive(self, value, atrb_name):
        if atrb_name == "batch_size" and value is None:
            return value

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{atrb_name} must be of type : int or float, got {type(value).__name__}"
            )
        if value <= 0:
            raise ValueError(f"{atrb_name} must be a positive value")
        return value

    def _validate_penalty(self, value):
        penalty_list = ["l1", "l2"]
        if value is None:
            return None

        if not isinstance(value, (str)):
            raise TypeError(
                f"Penalty must be a string or None, got {type(value).__name__}"
            )

        normlized_value = value.strip().lower()
        if normlized_value not in penalty_list:
            raise ValueError(f"{normlized_value} is not a valid penalty type")
        return normlized_value

    def _sigmoid_function(self, X):
        logit = X @ self.beta + self.intercept

        P_x = 1 / (np.exp(-logit) + 1)
        return P_x

    def fit(self, X, Y, n):
        p = X.shape[1]
        self.beta = np.zeros((p, 1))
        self.intercept = 0
        prev_loss = float("inf")
        for _ in range(self.n_itr):
            if self.batch_size is None:
                X_GD, Y_GD = X, Y
                print(self.batch_size)
            else:
                idx = np.random.choice(n, size=self.batch_size, replace=False)
                X_GD = X[idx]
                Y_GD = Y[idx].reshape(-1, 1)

            loss = self.cross_entropy_loss(Y, X)
            self.losses.append(loss)

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
            if self.penalty == "l2":
                regularization_l2 = (1 / self.C) * self.beta
                gradient_cel = (1 / len(Y_GD)) * (
                    (X_GD.T @ (self._sigmoid_function(X_GD) - Y_GD)) + regularization_l2
                )
            elif self.penalty == "l1":
                regularization_l1 = (1 / self.C) * np.sign(self.beta)
                gradient_cel = (1 / len(Y_GD)) * (
                    (X_GD.T @ (self._sigmoid_function(X_GD) - Y_GD)) + regularization_l1
                )

            else:
                gradient_cel = (1 / len(Y_GD)) * (
                    X_GD.T @ (self._sigmoid_function(X_GD) - Y_GD)
                )
            gradient_intercept = (1 / len(Y_GD)) * np.sum(
                self._sigmoid_function(X_GD) - Y_GD
            )
            self.beta = self.beta - (self.lr * gradient_cel)
            self.intercept = self.intercept - (self.lr * gradient_intercept)

        return self.beta, self.losses

    def cross_entropy_loss(self, Y, X):
        P_x = self._sigmoid_function(X)
        eps = 1e-15

        P_x = np.clip(P_x, eps, 1 - eps)
        cost_fn = -np.mean(Y.T @ np.log(P_x) + (1 - Y).T @ np.log(1 - P_x))

        return cost_fn

    def predict(self, X, threshold):
        predicted_probabilities = self._sigmoid_function(
            X
        )  # P(Y_hat) while Y_hat = Xbeta
        predicted_class = (predicted_probabilities >= threshold).astype(int)
        return predicted_class, predicted_probabilities

    def accuracy(self, Y, Y_pred):
        accuracy = np.mean(Y_pred == Y)
        return accuracy


def generate_dummy_data(n=1000, p=3, seed=1):
    np.random.seed(seed)

    X = np.random.randn(n, p)

    coeff_true = np.array([2] * (p))  # (p,)
    intercept_true = 3.0  #
    P_x = np.exp(X @ coeff_true + intercept_true) / (
        1 + np.exp(X @ coeff_true + intercept_true)
    )

    response = np.random.binomial(1, P_x)

    return response, coeff_true, intercept_true, X


def sigmoid_function(X, beta):
    logit = X @ beta
    P_x = 1 / (np.exp(-logit) + 1)
    return P_x


def cross_entropy_loss(X, y, beta):
    sigmoid_fn = sigmoid_function(X, beta)

    # clipping the array to avoid log(0)
    eps = 1e-15
    sigmoid_fn = np.clip(sigmoid_fn, eps, 1 - eps)
    cost_fn = -np.mean(y.T @ np.log(sigmoid_fn) + (1 - y).T @ np.log(1 - sigmoid_fn))
    return cost_fn


def stochastic_gradient_decent(lr, n_itr, Y, p, X, n):
    beta_est = np.zeros((p, 1))
    losses = []
    for i in range(n_itr):
        # Randomly selecting (x_i,y_i)
        idx = np.random.choice(n, size=1, replace=False)
        Xs = X[idx]
        Ys = Y[idx].reshape(-1, 1)

        # Calculting P(X)
        sigmoid_fn = sigmoid_function(Xs, beta_est)

        # calculating the gradient for the cross entropy loss
        gradient_cel = Xs.T @ (sigmoid_fn - Ys)
        beta_est_new = beta_est - (lr * gradient_cel)

        current_loss = cross_entropy_loss(X, Y, beta_est)
        losses.append(current_loss)

        if np.linalg.norm(beta_est_new - beta_est) < 1e-6:
            break

        beta_est = beta_est_new

    return beta_est, losses


def mini_batch_gradient_decent(lr, n_itr, batch_size, Y, p, X, n):
    beta_est = np.zeros((p, 1))
    losses = []

    for i in range(n_itr):
        idx = np.random.choice(n, size=batch_size, replace=False)
        Xb = X[idx]
        Yb = Y[idx].reshape(-1, 1)

        sigmoid_fn = sigmoid_function(Xb, beta_est)

        gradient_cel = (1 / batch_size) * (Xb.T @ (sigmoid_fn - Yb))

        beta_est_new = beta_est - (lr * gradient_cel)
        current_loss = cross_entropy_loss(X, Y, beta_est)
        losses.append(current_loss)
        if np.linalg.norm(beta_est_new - beta_est) < 1e-6:
            break
        beta_est = beta_est_new
    return beta_est, losses


def predict_proba(X, beta):
    return sigmoid_function(X, beta)


def predict(X, beta, threshold):
    predicted_prob = predict_proba(X, beta)
    predicted_class = (predicted_prob >= threshold).astype(int)
    return predicted_class, predicted_prob


def accuracy(y, pred_y):
    return np.mean(y == pred_y)


Y, beta_true, intercept_true, X = generate_dummy_data(n=10000, p=10, seed=10)

model = LogisticRegression(
    learning_rate=0.1, n_itr=20000, batch_size=100, C=2, penalty="L2"
)
beta_class, loss_class = model.fit(X, Y, 1000)
predicted_class_cl, predicted_prob_cl = model.predict(X, 0.5)
accuracy_class = model.accuracy(Y, predicted_class_cl)

print(beta_class)
print(beta_true)
