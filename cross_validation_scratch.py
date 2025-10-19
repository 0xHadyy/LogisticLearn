import numpy as np


# K-Fold method
# Statified K-Fold method
# score


class CrossValidation:
    def __init__(
        self, model=None, X=None, y=None, cv=5, score=None, stratified=False, seed=67
    ):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv
        self.score = score
        self.stratified = stratified
        self.seed = seed
        self.folds = []

    def validate_input(self):
        pass

    def validate_str(self):
        pass

    def _split(self, fold_size, n, i):
        start = i * fold_size
        end = (i + 1) * fold_size
        test_idx = n[start:end]
        train_idx = np.concatenate((n[:start], n[end:]))

        return test_idx, train_idx

    def _strartify(self):
        class_idx = {}
        for class_label in np.unique(self.y):
            class_idx[class_label] = np.where(self.y == class_label)[0]

        folds = [([], []) for _ in range(self.cv)]

        # Looping over each class
        for class_label, idx in class_idx.items():
            class_samples = len(idx)
            rng = np.random.default_rng(self.seed)
            rng.shuffle(idx)

            fold_size = class_samples / self.cv
            # Split the strata into tran / test
            for i in range(self.cv):
                test_idx, train_idx = self._split(fold_size, class_samples, i)

                folds[i][0].extend(train_idx)
                folds[i][1].extend(test_idx)

        folds = ((np.array(train), np.array(test)) for train, test in folds)

    def _KFold(self):
        n_size = self.X.shape[0]
        n = np.arange(n_size)
        # shuffle
        rng = np.random.default_rng(self.seed)
        rng.shuffle(n)

        fold_size = len(n) // self.cv

        for i in range(self.cv):
            test_idx, train_idx = self._split(fold_size, n, i)

            self.folds.append((test_idx, train_idx))
            print(self.folds)
