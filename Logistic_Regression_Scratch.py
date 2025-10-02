import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=100)


class LogisticRegression:
    def __init__(
        self,
        penalty=None,
        cv=5,
        lr=0.001,
        n_itr=2000,
        batch_size=None,
        tol=1e-9,
        C=1.0,
    ):
        self.lr = self._validate_positive(lr, "learning_rate")
        self.cv = cv
        self.n_itr = int(self._validate_positive(n_itr, "n_itr"))
        self.batch_size = self._validate_positive(batch_size, "batch_size")
        self.penalty = self._validate_penalty(penalty)
        self.tol = self._validate_positive(tol, "tol")
        self.C = self._validate_positive(C, "C")
        self.beta = None
        self.intercept = 0
        self.losses = []

    def get_params(self):
        return {
            "penalty": self.penalty,
            "lr": self.lr,
            "n_itr": self.n_itr,
            "batch_size": self.batch_size,
            "tol": self.tol,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

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
        P_x = np.where(
            logit >= 0, 1 / (1 + np.exp(-logit)), np.exp(logit) / (1 + np.exp(logit))
        )
        return P_x

    def fit(self, X, Y):
        p = X.shape[1]
        n = X.shape[0]
        self.beta = np.zeros((p, 1))
        self.intercept = 0
        prev_loss = float("inf")
        for _ in range(self.n_itr):
            if self.batch_size is None:
                # X_GD, Y_GD = X, Y
                X_GD = X
                Y_GD = Y.reshape(-1, 1)
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
        cost_fn = -np.mean(Y.T * np.log(P_x) + (1 - Y).T * np.log(1 - P_x))

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


def clone(estimator):
    # Know the class of the estimator
    cls = estimator.__class__
    params = estimator.get_params()

    return cls(**params)


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


def cross_validate(model, X, Y, seed, shuffle=True, cv=5):
    n = X.shape[0]
    if cv > n:
        raise ValueError(f"Number of folds {cv} with {n} samples ")
    losses = []
    folds = K_folds(n, cv, shuffle, seed)
    # folds = stratified_K_folds(n, cv, Y, shuffle, seed)
    for train_idx, test_idx in folds:
        # clone the model
        model_cv = clone(model)

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        model_cv.fit(X_train, Y_train)
        loss = model_cv.cross_entropy_loss(Y_test, X_test)
        losses.append(loss)

    return losses, np.mean(losses)


def K_folds(n, K, shuffle, seed):
    n = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(n)
    folds = []
    fold_size = len(n) // K
    print(fold_size)
    for i in range(K):
        start = i * fold_size
        # when n %K !=0 the last fold takes all the reminder
        end = (i + 1) * fold_size if i < K - 1 else len(n)
        test_idx = n[start:end]
        train_idx = np.concatenate([n[:start], n[end:]])
        folds.append((train_idx, test_idx))

    return folds


def stratified_K_folds(n, K, y, shuffle, seed):
    # order each class
    class_idx = {}
    for class_label in np.unique(y):
        # [0] to select the array in the np.where return tuple
        class_idx[class_label] = np.where(y == class_label)[0]

    folds = [([], []) for _ in range(K)]
    # Create Strata
    for class_label, idx in class_idx.items():
        class_samples = len(idx)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)

        fold_size_class = class_samples // K
        rest = class_samples % K

        for i in range(K):
            start = i * fold_size_class
            end = (i + 1) * fold_size_class + (1 if i < rest else 0)
            test_idx_class = idx[start:end]
            train_idx_class = np.concatenate([idx[:start], idx[end:]])
            # setdiff1d no need to extend for train idx
            folds[i][0].extend(train_idx_class)
            folds[i][1].extend(test_idx_class)

    folds = [(np.array(train), np.array(test)) for train, test in folds]
    return folds


Y, beta_true, intercept_true, X = generate_dummy_data(n=200, p=3, seed=10)

model = LogisticRegression(lr=0.1, n_itr=800, batch_size=None, C=1, penalty=None)

beta, loss_train = model.fit(X, Y)

losses, avg_loss = cross_validate(model, X, Y, seed=100, shuffle=True, cv=10)

predicted_class_cl, predicted_prob_cl = model.predict(X, 0.5)
accuracy_class = model.accuracy(Y, predicted_class_cl)

print(beta)
# print(model_clone
print(avg_loss)
