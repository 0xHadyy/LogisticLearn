import numpy as np
import random


np.set_printoptions(precision=3, suppress=True, linewidth=100)


class LogisticRegression:
    def __init__(
        self,
        penalty= None,
        learning_rate=0.001,
        n_itr=2000,
        batch_size=None,
    ):
        self.lr = learning_rate
        self.n_itr = n_itr
        self.batch_size = batch_size
        self.tol = 1e-6
        self.beta = None
        self.losses = []

    def _sigmoid_function(self, X):
        logit = X @ self.beta

        P_x = 1 / (np.exp(-logit) + 1)
        return P_x

    def fit(self, X, Y, n):
        p = X.shape[1]
        self.beta = np.zeros((p, 1))
        prev_loss = float("inf")
        for _ in range(self.n_itr):
            if self.batch_size is None:
                X_GD, Y_GD = X, Y
            else:
                idx = np.random.choice(n, size=self.batch_size, replace=False)
                X_GD = X[idx]
                Y_GD = Y[idx].reshape(-1, 1)

            loss = self.cross_entropy_loss(Y, X)
            self.losses.append(loss)

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

            gradient_cel = (1 / len(Y_GD)) * (
                X_GD.T @ (self._sigmoid_function(X_GD) - Y_GD)
            )

            self.beta = self.beta - (self.lr * gradient_cel)

        return self.beta, self.losses

    def cross_entropy_loss(self, Y, X):
        P_x = self._sigmoid_function(
            X,
        )
        eps = 1e-15

        P_x = np.clip(P_x, eps, 1 - eps)
        cost_fn = -np.mean(Y.T @ np.log(P_x) + (1 - Y).T @ np.log(1 - P_x))

        return cost_fn

    def predict(self, X, thershold):
        predicted_probabilities = self._sigmoid_function(
            X
        )  # P(Y_hat) while Y_hat = Xbeta
        predicted_class = (predicted_probabilities >= thershold).astype(int)
        return predicted_class, predicted_probabilities

    def accuracy(self, Y, Y_pred):
        accuracy = np.mean(Y_pred == Y)
        return accuracy

    def 


def generate_dummy_data(n=1000, p=3, seed=1):
    np.random.seed(seed)

    X_raw = np.random.randn(n, p - 1)
    intercept = np.ones((n, 1))

    X = np.hstack([intercept, X_raw])

    coeff_true = np.array([3] + [2] * (p - 1))

    P_x = np.exp(X @ coeff_true) / (1 + np.exp(X @ coeff_true))

    response = np.random.binomial(1, P_x)

    return response, coeff_true, X


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


Y, beta_true, X = generate_dummy_data()

model = LogisticRegression(learning_rate=0.1, n_itr=2000, batch_size=100)
beta_class, loss_class = model.fit(X, Y, 1000)
predicted_class_cl, predicted_prob_cl = model.predict(X, 0.5)
accuracy_class = model.accuracy(Y, predicted_class_cl)

beta_SDG, losses = stochastic_gradient_decent(0.1, 2000, Y, 3, X, 1000)

beta_MBGD, losss_mb = mini_batch_gradient_decent(0.1, 2000, 1000, Y, 3, X, 1000)

predicted_class, predicted_prob = predict(X, beta_MBGD, 0.5)

accuracy = accuracy(Y, predicted_class)

print(accuracy_class)
print(accuracy)
