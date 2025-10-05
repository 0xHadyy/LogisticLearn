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
        tol=1e-6,
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
            "C": self.C,
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
        logit = np.clip(logit, -500, 500)

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

            if self.penalty == "l2":
                regularization = (1 / (self.C * 2)) * self.beta
            elif self.penalty == "l1":
                regularization = (1 / self.C) * np.sign(self.beta)
            else:
                regularization = 0

            gradient_cel = (1 / len(Y_GD)) * (
                X_GD.T @ (self._sigmoid_function(X_GD) - Y_GD)
            ) + regularization

            gradient_intercept = (1 / len(Y_GD)) * np.sum(
                self._sigmoid_function(X_GD) - Y_GD
            )
            self.beta = self.beta - (self.lr * gradient_cel)
            # print(self.beta)
            self.intercept = self.intercept - (self.lr * gradient_intercept)

            loss = self.cross_entropy_loss(Y, X)
            self.losses.append(loss)

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return self.beta, self.losses

    def cross_entropy_loss(self, Y, X):
        Y = Y.reshape(-1, 1)
        P_x = self._sigmoid_function(X)
        eps = 1e-15

        P_x = np.clip(P_x, eps, 1 - eps)
        reg_loss = 0
        cost_fn = -np.mean(Y * np.log(P_x) + (1 - Y) * np.log(1 - P_x))
        if self.beta is not None:
            if self.penalty == "l2":
                reg_loss = (1 / (self.C * 2)) * np.sum(self.beta**2)
            elif self.penalty == "l1":
                reg_loss = (1 / self.C) * np.sum(np.abs(self.beta))
        return cost_fn + reg_loss

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
    # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    coeff_true = np.array([2] * (p))  # (p,)
    intercept_true = 3.0  #
    P_x = 1 / (1 + np.exp(-(X @ coeff_true + intercept_true)))

    response = np.random.binomial(1, P_x)

    return response, coeff_true, intercept_true, X


def cross_validate(model, X, Y, seed=10, shuffle=True, cv=5):
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


def grid_searchCV(model, param_grid, cv, X, Y):
    lr_values = param_grid.get("lr_values")
    C_values = param_grid.get("C_values")
    print(lr_values)
    print(C_values)

    score_cv = {}
    best_params = None  # best coefficients
    best_estimator = None  # best model
    best_score = np.inf  # best CV score

    for lr_value in lr_values:
        for C_value in C_values:
            model_CV = clone(model)
            model_CV.C = C_value
            model_CV.lr = lr_value
            print(f"the value of lr is :{lr_value} the value of C is : {C_value} ")
            loss, avg_loss = cross_validate(model_CV, X, Y, cv=cv)
            score_cv[(lr_value, C_value)] = avg_loss
            print(f"{avg_loss}< {best_score}")
            if avg_loss < best_score:
                best_score = avg_loss
                best_estimator = model_CV
                best_params = model_CV.get_params()
                print(best_params)
        # print(f"for C={C_value} and lr={lr_value} the avg loss of the model is ")

        #  print(avg_loss)
        #  print("---------")

    # Train the best model on the full data set
    print("the best score is :", best_score)
    print("the best params are : ", best_params)
    if (best_estimator is not None) and (best_params is not None):
        best_estimator.lr = best_params["lr"]
        best_estimator.C = best_params["C"]
        beta, loss = best_estimator.fit(X, Y)
        print(f"the best beta is {beta}")
    return {
        "best_score": best_score,
        "best_model": best_estimator,
        "best_params": best_params,
        "score_cv": score_cv,
    }


Y, beta_true, intercept_true, X = generate_dummy_data(n=300, p=3, seed=24)

model = LogisticRegression(lr=0.1, n_itr=900, batch_size=30, C=0.01, penalty=None)
beta, loss = model.fit(X, Y)
losses, avg_loss = cross_validate(model, X, Y, seed=10, shuffle=True, cv=10)
param_grid = {
    "lr_values": [0.1, 0.01, 0.001],
    "C_values": [0.01, 0.1, 0.2, 0.3],
}
score = grid_searchCV(model, param_grid, 10, X, Y)
modelCV = score["best_model"]
# beta, loss_train = model.fit(X, Y)
# print(beta)

# model = score["best_model"]


print(f"the best (base model) betas are : {beta}")
print(f"the base model avg loss is : {avg_loss}")
# print(f"the best model avg_loss is {avg_loss}")
# predicted_class_cl, predicted_prob_cl = model.predict(X, 0.5)
# accuracy_class = model.accuracy(Y, predicted_class_cl)
# print(accuracy_class)
