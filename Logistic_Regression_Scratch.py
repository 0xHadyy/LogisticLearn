import numpy as np
import random


np.set_printoptions(precision=3, suppress=True, linewidth=100)


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
    predicted_class = np.empty_like(predicted_prob)
    for i in range(len(predicted_prob)):
        if predicted_prob[i] >= threshold:
            predicted_class[i] = 1
        else:
            predicted_class[i] = 0
    return predicted_class, predicted_prob


def accuracy(y, pred_y):
    return np.mean(y == pred_y)


Y, beta_true, X = generate_dummy_data()

beta_SDG, losses = stochastic_gradient_decent(0.1, 2000, Y, 3, X, 1000)

beta_MBGD, losss_mb = mini_batch_gradient_decent(0.1, 2000, 1000, Y, 3, X, 1000)

predicted_class, predicted_prob = predict(X, beta_SDG, 0.5)

accuracy = accuracy(Y, predicted_class)

print(predicted_class)

print(accuracy)
