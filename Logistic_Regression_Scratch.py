import numpy as np
from sklearn import datasets
from sklearn.metrics import f1_score


np.set_printoptions(precision=3, suppress=True, linewidth=100)


class LogisticRegression:
    def __init__(
        self,
        penalty=None,
        lr=0.001,
        n_itr=2000,
        batch_size=None,
        tol=1e-6,
        C=1.0,
    ):
        self.lr = self._validate_positive(lr, "learning_rate")
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
        # print(f"the value of beta is {self.beta} ")
        # print(f"the shape of X is :{X.shape}, beta :{self.beta.shape}")
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

            p = self._sigmoid_function(X_GD)
            gradient_cel = (X_GD.T @ (p - Y_GD)) / len(Y_GD)
            gradient_intercept = np.sum(p - Y_GD) / len(Y_GD)

            if self.penalty == "l2":
                regularization = (1.0 / self.C) * self.beta
                gradient_cel = gradient_cel + regularization
                grad_norm = np.linalg.norm(gradient_cel)
                if grad_norm > 1e3:
                    gradient_cel = gradient_cel / grad_norm * 1e3
                self.beta = self.beta - (self.lr * gradient_cel)

            elif self.penalty == "l1":
                beta_temp = self.beta - (self.lr * gradient_cel)
                threshold = self.lr * (1.0 / self.C)
                self.beta = np.sign(beta_temp) * np.maximum(
                    np.abs(beta_temp) - threshold, 0
                )

            else:
                self.beta = self.beta - (self.lr * gradient_cel)

            self.intercept = self.intercept - (self.lr * gradient_intercept)

            loss = self.cross_entropy_loss(Y_GD, X_GD)
            self.losses.append(loss)

            # early break
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return self.beta, self.losses

    def cross_entropy_loss(self, Y, X):
        Y = Y.reshape(-1, 1)
        P_x = self._sigmoid_function(X)
        eps = 1e-15

        # Clipping small values
        P_x = np.clip(P_x, eps, 1 - eps)
        reg_loss = 0
        cost_fn = -np.mean(Y * np.log(P_x) + (1 - Y) * np.log(1 - P_x))

        if self.beta is not None:
            if self.penalty == "l2":
                reg_loss = (1 / (self.C * 2)) * np.sum(self.beta**2)
            elif self.penalty == "l1":
                reg_loss = (1 / self.C) * np.sum(np.abs(self.beta))

        loss = cost_fn + reg_loss

        return loss

    def predict(self, X, threshold):
        # use unseen data X
        predicted_probabilities = self._sigmoid_function(
            X
        )  # P(Y_hat) while Y_hat = Xbeta
        # print(f"the predictd probability is {predicted_probabilities}")

        predicted_class = (predicted_probabilities >= threshold).astype(int)
        return predicted_class, predicted_probabilities

    def accuracy(self, Y, Y_pred):
        accuracy = np.mean(Y_pred == Y.reshape(-1, 1))
        return accuracy


def clone(estimator):
    # Know the class of the estimator
    cls = estimator.__class__
    params = estimator.get_params()
    return cls(**params)


def generate_challenging_data(n=1000, p=20, seed=1, noise_level=0.3):
    np.random.seed(seed)

    # Create correlated features (multicollinearity)
    X = np.random.randn(n, p)
    # Add correlation between features
    for i in range(1, p):
        X[:, i] += 0.7 * X[:, i - 1]  # High correlation

    # True model: only first 5 features matter
    coeff_true = np.zeros(p)
    coeff_true[:5] = np.random.uniform(-2, 2, 5)
    coeff_true[5:] = 0  # Truly sparse

    print("the true coeff are : ", coeff_true)
    intercept_true = np.random.uniform(-1, 1)

    print("the true intercept :", intercept_true)
    logit = X @ coeff_true + intercept_true
    # Add more noise to make it harder
    logit += np.random.normal(0, noise_level, n)

    P_x = 1 / (1 + np.exp(-logit))
    response = np.random.binomial(1, P_x)

    return response, coeff_true, intercept_true, X


def generate_dummy_data(n=1000, p=3, seed=1):
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    X = np.random.randn(n, p)
    noise = rng.normal(0, 0.2, size=X.shape)
    X += noise
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    coeff_true = rng.uniform(-3, 2, size=p)
    print("the true coeff are : ", coeff_true)
    intercept_true = rng.uniform(-1, 1)
    print("the true intercept :", intercept_true)

    logit = X @ coeff_true + intercept_true
    P_x = 1 / (1 + np.exp(-logit))

    response = np.random.binomial(1, P_x)

    return response, coeff_true, intercept_true, X


def pure_loss(model, X, Y):
    P = model._sigmoid_function(X)
    P = np.clip(P, 1e-15, 1 - 1e-15)
    Y = Y.reshape(-1, 1)
    loss_pure = -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))
    return loss_pure


def cross_validate(model, X, Y, seed=10, shuffle=True, cv=10):
    n = X.shape[0]
    if cv > n:
        raise ValueError(f"Number of folds {cv} with {n} samples ")
    losses = []
    # folds = K_folds(n, cv, shuffle, seed)
    folds = stratified_K_folds(cv, Y, shuffle, seed)
    for train_idx, test_idx in folds:
        # clone the model
        model_cv = clone(model)

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        model_cv.fit(X_train, Y_train)
        loss = model_cv.cross_entropy_loss(Y_test, X_test)
        if model_cv.penalty is not None:
            loss = pure_loss(model_cv, X_test, Y_test)
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


def stratified_K_folds(K, y, shuffle, seed):
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

        # Splitting each class strata into train/test fold
        for i in range(K):
            start = i * fold_size_class
            end = (i + 1) * fold_size_class + (1 if i < rest else 0)
            test_idx_class = idx[start:end]
            train_idx_class = np.concatenate([idx[:start], idx[end:]])
            # setdiff1d no need to extend for train idx
            # collecting all the test folds from each strat into => THE TEST FOLD
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
            model_CV.n_itr = model.n_itr
            model_CV.batch_size = model.batch_size
            # print(f"the value of lr is :{lr_value} the value of C is : {C_value} ")
            loss, avg_loss = cross_validate(model_CV, X, Y, cv=cv)

            score_cv[(lr_value, C_value)] = avg_loss
            # print(f"{avg_loss}< {best_score}")
            if avg_loss < best_score:
                best_score = avg_loss
                best_estimator = model_CV
                best_params = model_CV.get_params()
                # print(best_params)
        # print(f"for C={C_value} and lr={lr_value} the avg loss of the model is ")

        #  print(avg_loss)
        #  print("---------")

    # Train the best model on the full data set
    print("========gridsearch model=====")
    print("the best params are : ", best_params)
    if (best_estimator is not None) and (best_params is not None):
        print(f"the params are : {best_estimator.lr} and {best_estimator.C}")
        _, avg_loss = cross_validate(best_estimator, X, Y, seed=2, shuffle=True, cv=10)
        best_estimator.fit(X, Y)
        print(f"the best beta is {best_estimator.beta}")
        print("the avg loss right after fitting is : ", avg_loss)

    return {
        "best_score": best_score,
        "best_model": best_estimator,
        "best_params": best_params,
        "score_cv": score_cv,
    }


def load_breast_cancer():
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    # Manual standardization
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X, y, data.feature_names


X, Y, name = load_breast_cancer()
# Y, beta_true, intercept_true, X = generate_challenging_data(n=600, p=6, seed=2)


model = LogisticRegression(lr=0.1, n_itr=10000, batch_size=50, penalty=None)


beta, _ = model.fit(X, Y)
losses, avg_loss = cross_validate(model, X, Y, seed=2, shuffle=True, cv=10)
print(f"the best model avg_loss is {avg_loss}")
predicted_class_cl, predicted_prob_cl = model.predict(X, 0.5)
accuracy_class = model.accuracy(Y, predicted_class_cl)
print("=============Base model-======= ")
print("the avg loss", avg_loss)
print("the estimated coeffi", beta)
print("the estimated intercept is ", model.intercept)
print("the accuracy is :", accuracy_class)
print(model.get_params())

model_l2 = LogisticRegression(lr=0.1, n_itr=10000, batch_size=50, penalty="l2", C=10)
beta_l2, _ = model_l2.fit(X, Y)
_, avg_loss_l2 = cross_validate(model_l2, X, Y, seed=2, shuffle=True, cv=10)
predicted_class_l2, predicted_prob_l2 = model_l2.predict(X, 0.5)
accuracy_class_l2 = model_l2.accuracy(Y, predicted_class_l2)
print("=============L2 model-======= ")
print("the avg loss ", avg_loss_l2)
print("the estimated coeffi", beta_l2)
print("the estimated intercept is ", model_l2.intercept)
print("the accuracy is :", accuracy_class_l2)
print(model_l2.get_params())


model_l1 = LogisticRegression(lr=0.1, n_itr=10000, batch_size=50, penalty="l1", C=10)
beta_l1, _ = model_l1.fit(X, Y)
_, avg_loss_l1 = cross_validate(model_l1, X, Y, seed=2, shuffle=True, cv=10)
predicted_class_l1, predicted_prob_l1 = model_l1.predict(X, 0.5)
accuracy_class_l1 = model_l1.accuracy(Y, predicted_class_l1)
print("=============L1 model-======= ")
print("the avg loss ", avg_loss_l1)
print("the estimated coeffi", beta_l1)
print("the estimated intercept is ", model_l1.intercept)
print("the accuracy is :", accuracy_class_l1)
print(model_l1.get_params())

param_grid = {
    "lr_values": [0.01, 0.001, 0.1],
    "C_values": [0.01, 0.1, 1, 10, 100, 1000],
}
score = grid_searchCV(model_l2, param_grid, 10, X, Y)
modelCV = score["best_model"]
print(score["best_params"])
losses_best, avg_loss_best = cross_validate(modelCV, X, Y, seed=2, shuffle=True, cv=10)
predicted_class_best, predicted_prob_best = modelCV.predict(X, 0.5)
accuracy_best = modelCV.accuracy(Y, predicted_class_best)
print(f"the avg_loss for the grid search model i {avg_loss_best}")
print("the accuracy of gridsearch fitted model is :", accuracy_best)
