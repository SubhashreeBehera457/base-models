class SimpleLinearRegression:
    """
    Simple Linear Regression implemented from scratch using
    Ordinary Least Squares (OLS).

    Fits a line y = coef_ * x + intercept_ by minimizing the
    sum of squared errors between predicted and actual values.
    """

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        """
        Compute coef_ (slope) and intercept_ using the closed-form
        OLS solution.

        Parameters
        ----------
        X_train : array-like of shape (n_samples,)
            Training feature values.
        y_train : array-like of shape (n_samples,)
            Training target values.

        Returns
        -------
        None
            Sets self.coef_ and self.intercept_ as side effects.
        """
        numerator = 0
        denomenator = 0

        for i in range(X_train.shape[0]):
            numerator = numerator + ((X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean()))
            denomenator = denomenator + (X_train[i] - X_train.mean())**2

        self.coef_ = numerator / denomenator
        self.intercept_ = y_train.mean() - self.coef_ * X_train.mean()

        print("Coef_ : ", self.coef_)
        print("intercept_ : ", self.intercept_)

    def predict(self, X_test):
        """
        Predict target values for given input features using the
        fitted line y = coef_ * x + intercept_.

        Parameters
        ----------
        X_test : array-like of shape (n_samples,)
            Input feature values to predict on.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted target values.
        """
        y_pred = (self.coef_ * X_test) + self.intercept_
        return y_pred
