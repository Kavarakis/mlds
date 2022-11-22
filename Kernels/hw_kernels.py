from calendar import c
import numpy as np
from cvxopt import solvers
from cvxopt import matrix
import matplotlib.pyplot as plt
import pandas as pd

EPSILON = 1e-5


class Linear:
    """An example of a kernel."""
    def __init__(self):
        # here a kernel could set its parameters
        pass

    # a-xprim b-x

    def __call__(self, A, B):
        """Can be called with vectors or matrices, see the
        comment for test_kernel"""
        return A.dot(B.T)


class Polynomial:
    def __init__(self, M):
        self.M = M

    def __call__(self, x1, x2):
        factor = 1
        if len(x1.shape) == 2:
            factor = x1.shape[1]
        result = np.power(1 + np.dot(x1, x2.T) / factor, self.M)
        return result


class RBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x1, x2):
        x1_sum_axis = 0
        x2_sum_axis = 0
        if x1.ndim == 2 and x2.ndim == 1:
            x1, x2 = x2, x1
        if x1.ndim == 2:
            x1_sum_axis = 1
        if x2.ndim == 2:
            x2_sum_axis = 1
        distances = (x1**2).sum(axis=x1_sum_axis).reshape(
            -1, 1) + (x2**2).sum(axis=x2_sum_axis) - 2 * np.dot(x1, x2.T)
        return np.squeeze(np.exp(-1 * distances / (2 * self.sigma**2)))


class KernelizedRidgeRegression:
    def __init__(self, lambda_, kernel=Linear):
        self.k = kernel
        self.l = lambda_

    def gramm(self):
        # Kernelizing Gram Matrix
        def f2(xprim, x2):
            return self.k(xprim, x2)

        def f1(x1):
            return np.apply_along_axis(func1d=f2, arr=self.x, axis=1, x2=x1)

        return np.apply_along_axis(func1d=f1, arr=self.x, axis=1)

    def fit(self, x, y):

        self.x = x
        self.y = y
        f_size = self.x.shape[0]
        gr = self.gramm()
        # print(gr[0][:10])
        self.alpha = np.linalg.inv((gr + self.l * np.eye(f_size)))
        self.alpha = self.alpha.dot(self.y)
        self.beta = self.x.T.dot(self.alpha)

        return self

    def predict(self, xp):
        def kernel_res(xprim):
            return np.apply_along_axis(func1d=ker,
                                       axis=1,
                                       arr=self.x,
                                       xprim=xprim)

        def ker(x_i, xprim):
            return self.k(xprim, x_i)

        res = np.apply_along_axis(func1d=kernel_res, axis=1, arr=xp)
        yhat = res.dot(self.alpha)
        return yhat


class SVR:
    def __init__(self, kernel, lambda_, epsilon):
        self.kernel = kernel
        self.C = 1 / lambda_
        self.epsilon = epsilon
        self.K = None
        self.y = None
        self.b = None
        self.alpha = None
        self.X = None
        self.l = None
        self.B = None

    def __generate_B__(self, l):
        B = np.zeros(shape=(l, 2 * l))
        for i in range(0, l):
            B[i][2 * i] = 1
            B[i][2 * i + 1] = -1
        return B

    def __calculate_b__(self):
        residual = np.squeeze(
            -self.epsilon + self.y.T -
            np.matmul(np.matmul(self.B, self.alpha).T, self.K))
        first_logical = np.zeros(shape=self.l, dtype=bool)
        second_logical = np.zeros(shape=self.l, dtype=bool)
        for i in range(self.l):
            if self.alpha[2 * i] < self.C - EPSILON or self.alpha[2 * i +
                                                                  1] > EPSILON:
                first_logical[i] = True
            if self.alpha[2 * i] > EPSILON or self.alpha[2 * i +
                                                         1] < self.C - EPSILON:
                second_logical[i] = True
        first_limit = max(residual[first_logical])
        second_limit = min((residual + 2 * self.epsilon)[second_logical])
        return (first_limit + second_limit) / 2

    def fit(self, X, y):
        self.l = X.shape[0]
        self.X = X
        self.y = np.reshape(y, newshape=(self.l, 1))
        self.K = self.kernel(X, X)
        self.B = self.__generate_B__(self.l)
        P = np.matmul(self.B.T, np.matmul(self.K, self.B))
        q = self.epsilon * np.ones(shape=(2 * self.l, 1)) - np.matmul(
            self.B.T, self.y)
        G = np.vstack((np.eye(2 * self.l), -np.eye(2 * self.l)))
        h = np.vstack((self.C * np.ones(shape=(2 * self.l, 1)),
                       np.zeros(shape=(2 * self.l, 1))))
        A = np.matmul(np.ones(shape=(1, self.l)), self.B)
        b = np.zeros(shape=(1, 1))
        solvers.options['show_progress'] = False
        res = solvers.qp(matrix(P.astype('float')), matrix(q.astype('float')),
                         matrix(G.astype('float')), matrix(h.astype('float')),
                         matrix(A.astype('float')), matrix(b.astype('float')))
        self.alpha = np.array(res['x']).reshape((2 * self.l, 1))
        self.b = self.__calculate_b__()
        return self

    def predict(self, X):
        return np.squeeze(
            np.matmul(np.matmul(self.B, self.alpha).T, self.kernel(self.X, X))
            + self.b)

    def get_b(self):
        return self.b

    def get_alpha(self):
        return self.alpha.reshape((self.l, 2))


def RMSE(y_predicted, y_true):
    return np.sqrt(np.power(y_predicted - y_true, 2).mean())


def MSE(y_predicted, y_true):
    return np.power(y_predicted - y_true, 2).mean()


def get_cross_validation_score(X, y, k, model, seed):
    data = np.column_stack((X, y))
    np.random.seed(seed)
    np.random.shuffle(data)
    splits = np.array_split(data, k)
    scores = []
    for i, split in enumerate(splits):
        test = split
        train = np.row_stack(
            [splits[j] for j in range(0, len(splits)) if j != i])
        X_train = train[:, :-1]
        y_train = train[:, -1]
        X_test = test[:, :-1]
        y_test = test[:, -1]

        t = model.fit(X_train, y_train)
        y_predict = t.predict(X_test)
        scores.append(RMSE(y_predict, y_test))

    test_score = np.mean(scores)
    return test_score


def normalize_input_data(x):
    def normalize_row(row):
        diff = np.max(row) - np.min(row)
        return (row - np.min(row)) / diff

    return np.apply_along_axis(func1d=normalize_row, axis=0, arr=x)


def get_support_vector_indices(alphas):
    test = [
        i for i, x in enumerate(alphas) if x[0] > EPSILON or x[1] > EPSILON
    ]
    return test


def bias_variance_estimate(estimator,
                           X_train,
                           y_train,
                           X_test,
                           y_test,
                           bootstrap_rounds=100):
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)

    # initialize dataframe for storing predictions on test data
    preds_test = pd.DataFrame(index=y_test.index)
    # for each round: draw bootstrap indices, train model on bootstrap data and make predictions on test data
    for r in range(bootstrap_rounds):
        boot = np.random.randint(len(y_train), size=len(y_train))
        preds_test[f'Model {r}'] = estimator.fit(
            X_train.iloc[boot, :].values,
            y_train.iloc[boot].values).predict(X_test.values)

    # calculate "average model"'s predictions
    mean_pred_test = preds_test.mean(axis=1)
    # compute and return: mse, squared bias and variance
    mse = preds_test.apply(
        lambda pred_test: MSE(y_test.values, pred_test.values)).mean()

    bias_squared = MSE(y_test.values, mean_pred_test.values)

    variance = preds_test.apply(
        lambda pred_test: MSE(mean_pred_test.values, pred_test.values)).mean()

    return mse, bias_squared, variance


def housing():
    h = pd.read_csv("housing2r.csv").to_numpy()
    test = h[round(len(h) * 0.8):, :]
    train = h[:-round(len(h) * 0.2), :]

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    X_train = normalize_input_data(X_train)
    X_test = normalize_input_data(X_test)

    M_values = np.array((range(1, 11)))
    sigma_values = np.arange(0.2, stop=3, step=0.2)

    # Poly
    lambda_values = np.array([
        np.exp(-30),
        np.exp(-20),
        np.exp(-10),
        np.exp(-5), 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50
    ])
    default_lambda_rmse_svr = []
    optimal_lambda_rmse_svr = []
    default_lambda_rmse_krr = []
    optimal_lambda_rmse_krr = []
    default_lambda_SVs = []
    optimal_lambda_SVs = []
    lambda_poly_svr = []
    lambda_poly_krr = []
    for M in M_values:
        lambda_scores_svr = []
        lambda_scores_krr = []

        for lambda_ in lambda_values:
            lambda_scores_svr.append(
                get_cross_validation_score(
                    X_train, y_train, 5,
                    SVR(kernel=Polynomial(M=M), lambda_=lambda_, epsilon=1),
                    0))
            lambda_scores_krr.append(
                get_cross_validation_score(
                    X_train, y_train, 5,
                    KernelizedRidgeRegression(kernel=Polynomial(M=M),
                                              lambda_=lambda_), 0))
        best_lambda_svr = lambda_values[np.argmin(lambda_scores_svr)]
        best_lambda_krr = lambda_values[np.argmin(lambda_scores_krr)]

        print(
            f"[SVR] For parameter M={M}, optimal lambda is {best_lambda_svr}")
        print(
            f"[KRR] For parameter M={M}, optimal lambda is {best_lambda_krr}")

        first_model_svr = SVR(kernel=Polynomial(M=M), lambda_=1,
                              epsilon=1).fit(X_train, y_train)
        second_model_svr = SVR(kernel=Polynomial(M=M),
                               lambda_=best_lambda_svr,
                               epsilon=1).fit(X_train, y_train)
        first_model_krr = KernelizedRidgeRegression(kernel=Polynomial(M=M),
                                                    lambda_=1).fit(
                                                        X_train, y_train)
        second_model_krr = KernelizedRidgeRegression(
            kernel=Polynomial(M=M),
            lambda_=best_lambda_krr).fit(X_train, y_train)
        default_lambda_SVs.append(
            len(get_support_vector_indices(first_model_svr.get_alpha())))
        optimal_lambda_SVs.append(
            len(get_support_vector_indices(second_model_svr.get_alpha())))

        default_lambda_rmse_svr.append(
            RMSE(first_model_svr.predict(X_test), y_test))
        optimal_lambda_rmse_svr.append(
            RMSE(second_model_svr.predict(X_test), y_test))

        default_lambda_rmse_krr.append(
            RMSE(first_model_krr.predict(X_test), y_test))
        optimal_lambda_rmse_krr.append(
            RMSE(second_model_krr.predict(X_test), y_test))
        lambda_poly_svr.append(best_lambda_svr)
        lambda_poly_krr.append(best_lambda_krr)
    plt.bar(M_values, lambda_poly_svr, label='SVR')
    plt.bar(M_values, lambda_poly_krr, label='KRR')
    print('Mean,med svr lambdas:', np.mean(lambda_poly_svr),
          np.median(lambda_poly_svr))
    print('Mean,med krr lambdas:', np.mean(lambda_poly_krr),
          np.median(lambda_poly_krr))

    plt.grid()
    plt.legend()
    plt.savefig('lambdas.png')

    fig, ax = plt.subplots()
    ax.plot(M_values, default_lambda_rmse_svr, 'ro-', label="Lambda = 1")
    ax.plot(M_values, optimal_lambda_rmse_svr, 'bo-', label="SVR - CV Lambda")

    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Value of M")
    ax.set_ylabel("RMSE")
    ax.set_title(
        "RMSE on Housing2r dataset \n using Support vector regression with Poly kernel"
    )
    for x, y, s in zip(M_values, default_lambda_rmse_svr, default_lambda_SVs):
        plt.text(x,
                 y,
                 s,
                 horizontalalignment='left',
                 verticalalignment='center',
                 fontdict={
                     'fontweight': 500,
                     'size': 12
                 })

    for x, y, s in zip(M_values, optimal_lambda_rmse_svr, optimal_lambda_SVs):
        plt.text(x,
                 y,
                 s,
                 horizontalalignment='left',
                 verticalalignment='center',
                 fontdict={
                     'fontweight': 500,
                     'size': 12
                 })
    plt.legend()
    plt.savefig("Poly_housing_fit_svr.png",
                facecolor='white',
                bbox_inches='tight')
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(M_values, default_lambda_rmse_krr, 'ro-', label="Lambda = 1")
    ax.plot(M_values, optimal_lambda_rmse_krr, 'bo-', label="KRR - CV Lambda")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Value of M")
    ax.set_ylabel("RMSE")
    ax.set_title(
        "RMSE on Housing2r dataset \n using Kernelized ridge regression with Poly kernel"
    )
    plt.legend()
    plt.savefig("Poly_housing_fit_krr.png",
                facecolor='white',
                bbox_inches='tight')
    fig.show()

    # RBF
    epsilon = 0.75
    default_lambda_rmse_rbf_svr = []
    optimal_lambda_rmse_rbf_svr = []
    default_lambda_rmse_rbf_krr = []
    optimal_lambda_rmse_rbf_krr = []
    default_lambda_SVs = []
    optimal_lambda_SVs = []
    for sigma in sigma_values:
        lambda_scores_svr = []
        lambda_scores_krr = []

        for lambda_ in lambda_values:
            lambda_scores_svr.append(
                get_cross_validation_score(
                    X_train, y_train, 5,
                    SVR(kernel=RBF(sigma=sigma),
                        lambda_=lambda_,
                        epsilon=epsilon), 0))
            lambda_scores_krr.append(
                get_cross_validation_score(
                    X_train, y_train, 5,
                    KernelizedRidgeRegression(kernel=RBF(sigma=sigma),
                                              lambda_=lambda_), 0))
        best_lambda = lambda_values[np.argmin(lambda_scores_svr)]
        print(
            f"[SVR] For parameter sigma={sigma}, optimal lambda is {best_lambda}"
        )
        first_model_svr = SVR(kernel=RBF(sigma=sigma),
                              lambda_=1,
                              epsilon=epsilon).fit(X_train, y_train)
        second_model_svr = SVR(kernel=RBF(sigma=sigma),
                               lambda_=best_lambda,
                               epsilon=epsilon).fit(X_train, y_train)

        first_model_krr = KernelizedRidgeRegression(kernel=RBF(sigma=sigma),
                                                    lambda_=1).fit(
                                                        X_train, y_train)
        second_model_krr = KernelizedRidgeRegression(kernel=RBF(sigma=sigma),
                                                     lambda_=best_lambda).fit(
                                                         X_train, y_train)

        default_lambda_SVs.append(
            len(get_support_vector_indices(first_model_svr.get_alpha())))
        optimal_lambda_SVs.append(
            len(get_support_vector_indices(second_model_svr.get_alpha())))

        default_lambda_rmse_rbf_svr.append(
            RMSE(first_model_svr.predict(X_test), y_test))
        optimal_lambda_rmse_rbf_svr.append(
            RMSE(second_model_svr.predict(X_test), y_test))

        default_lambda_rmse_rbf_krr.append(
            RMSE(first_model_krr.predict(X_test), y_test))
        optimal_lambda_rmse_rbf_krr.append(
            RMSE(second_model_krr.predict(X_test), y_test))

    fig, ax = plt.subplots()
    ax.plot(sigma_values,
            default_lambda_rmse_rbf_svr,
            'ro-',
            label="Lambda = 1")
    ax.plot(sigma_values,
            optimal_lambda_rmse_rbf_svr,
            'bo-',
            label="CV Lambda")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Value of sigma")
    ax.set_ylabel("RMSE")
    ax.set_title(
        f"RMSE, eps={epsilon} on Housing2r dataset \n using Support vector regression with RBF kernel"
    )
    for x, y, s in zip(sigma_values, default_lambda_rmse_rbf_svr,
                       default_lambda_SVs):
        plt.text(x,
                 y,
                 s,
                 horizontalalignment='left',
                 verticalalignment='center',
                 fontdict={
                     'fontweight': 500,
                     'size': 12
                 })

    for x, y, s in zip(sigma_values, optimal_lambda_rmse_rbf_svr,
                       optimal_lambda_SVs):
        plt.text(x,
                 y,
                 s,
                 horizontalalignment='left',
                 verticalalignment='center',
                 fontdict={
                     'fontweight': 500,
                     'size': 12
                 })
    plt.legend()
    plt.savefig(f"RBF_housing_fit_{epsilon}_svr.png",
                facecolor='white',
                bbox_inches='tight')
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(sigma_values,
            default_lambda_rmse_rbf_krr,
            'ro-',
            label="Lambda = 1")
    ax.plot(sigma_values,
            optimal_lambda_rmse_rbf_krr,
            'bo-',
            label="CV Lambda")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Value of sigma")
    ax.set_ylabel("RMSE")
    ax.set_title(
        f"RMSE, eps={epsilon} on Housing2r dataset using Kernelized ridge regression with RBF kernel"
    )
    plt.legend()
    plt.savefig(f"RBF_housing_fit_krr.png",
                facecolor='white',
                bbox_inches='tight')


def sine():
    _sine = pd.read_csv("sine.csv").to_numpy()

    possible_epsilon_values = [0.1, 0.5, 0.8, 1, 1.5, 2, 3]
    ## Sine
    X = _sine[:, :-1]
    y = _sine[:, 1:]

    X = normalize_input_data(X)

    rbf_svr = SVR(kernel=RBF(sigma=0.1), lambda_=0.1, epsilon=0.5).fit(X, y)
    poly_svr = SVR(kernel=Polynomial(M=12), lambda_=np.exp(-25),
                   epsilon=1).fit(X, y)
    rbf_krr = KernelizedRidgeRegression(kernel=RBF(sigma=0.1),
                                        lambda_=0.1).fit(X, y)
    poly_krr = KernelizedRidgeRegression(kernel=Polynomial(M=10),
                                         lambda_=np.exp(-20)).fit(X, y)
    # y_rbf = rbf.predict(X)
    # y_poly = poly.predict(X)

    to_plot_poly_svr = pd.DataFrame(np.array(
        [np.squeeze(X), np.squeeze(poly_svr.predict(X))]).T,
                                    columns=["x", "y"]).sort_values(by=["x"])
    to_plot_poly_krr = pd.DataFrame(np.array(
        [np.squeeze(X), np.squeeze(poly_krr.predict(X))]).T,
                                    columns=["x", "y"]).sort_values(by=["x"])
    to_plot_base = pd.DataFrame(np.array([np.squeeze(X),
                                          np.squeeze(y)]).T,
                                columns=["x", "y"]).sort_values(by=["x"])
    to_plot_rbf_svr = pd.DataFrame(np.array(
        [np.squeeze(X), np.squeeze(rbf_svr.predict(X))]).T,
                                   columns=["x", "y"]).sort_values(by=["x"])

    to_plot_rbf_krr = pd.DataFrame(np.array(
        [np.squeeze(X), np.squeeze(rbf_krr.predict(X))]).T,
                                   columns=["x", "y"]).sort_values(by=["x"])

    SV_rbf = to_plot_base.iloc[
        get_support_vector_indices(rbf_svr.get_alpha()), ]
    SV_poly = to_plot_base.iloc[
        get_support_vector_indices(poly_svr.get_alpha()), ]

    fig, ax = plt.subplots()
    ax.plot(to_plot_base["x"],
            to_plot_base["y"],
            "bo",
            markersize=1,
            alpha=1,
            label="Data points")
    ax.plot(to_plot_rbf_svr["x"],
            to_plot_rbf_svr["y"],
            "ro-",
            label="Fit - SVR",
            linewidth=1,
            markersize=1.5,
            color='g')
    ax.plot(to_plot_rbf_krr["x"],
            to_plot_rbf_krr["y"],
            "ro-",
            label="Fit - KRR",
            linewidth=1,
            markersize=1.5,
            color='y')
    ax.scatter(SV_rbf["x"],
               SV_rbf["y"],
               s=50,
               color="black",
               marker='+',
               label="Support vectors",
               alpha=1,
               linewidths=3)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SVR and KRR fit with RBF kernel to sine data")
    plt.legend()
    plt.savefig("RBF_sine_fit.png", facecolor='white', bbox_inches='tight')
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(to_plot_base["x"],
            to_plot_base["y"],
            "bo",
            markersize=1,
            label="Data points")
    ax.plot(to_plot_poly_svr["x"],
            to_plot_poly_svr["y"],
            "ro-",
            label="Fit - SVR ",
            linewidth=1,
            markersize=1.5,
            color='g')
    ax.plot(to_plot_poly_krr["x"],
            to_plot_poly_krr["y"],
            "ro-",
            label="Fit - KRR ",
            linewidth=1,
            markersize=1.5,
            color='y')

    ax.scatter(SV_poly["x"],
               SV_poly["y"],
               s=50,
               color="black",
               marker='+',
               label="Support vectors",
               linewidths=3)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SVR and KRR fit with Polynomial kernel to sine data")
    plt.legend()
    plt.savefig("Poly_sine_fit.png", facecolor='white', bbox_inches='tight')
    fig.show()
    plt.clf()

    print('KRR_POLY:', RMSE(poly_krr.predict(X), y))
    print('SVR_POLY:', RMSE(poly_svr.predict(X), y))
    print('KRR_RBF:', RMSE(rbf_krr.predict(X), y))
    print('SVR_RBF:', RMSE(rbf_svr.predict(X), y))


def analysis():
    h = pd.read_csv("housing2r.csv").to_numpy()
    test = h[round(len(h) * 0.8):, :]
    train = h[:-round(len(h) * 0.2), :]

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    X_train = normalize_input_data(X_train)
    X_test = normalize_input_data(X_test)
    print('HOUSING: \n')
    print(
        'BV decomp - SVR:',
        bias_variance_estimate(SVR(kernel=Polynomial(1), lambda_=1, epsilon=1),
                               X_train, y_train, X_test, y_test, 100))
    print(
        'BV decomp - KRR:',
        bias_variance_estimate(
            KernelizedRidgeRegression(kernel=Polynomial(1), lambda_=1),
            X_train, y_train, X_test, y_test, 100))
    print('SINE: \n')
    _sine = pd.read_csv("sine.csv").to_numpy()
    X = _sine[:, :-1]
    y = _sine[:, 1:]

    X = normalize_input_data(X)
    poly_svr = SVR(kernel=Polynomial(M=12), lambda_=np.exp(-25),
                   epsilon=1).fit(X, y)
    poly_krr = KernelizedRidgeRegression(kernel=Polynomial(M=10),
                                         lambda_=np.exp(-20)).fit(X, y)

    print('BV decomp - SVR:', bias_variance_estimate(poly_svr, X, y, X, y,
                                                     100))
    print('BV decomp - KRR:', bias_variance_estimate(poly_krr, X, y, X, y,
                                                     100))


if __name__ == "__main__":
    housing()
    sine()
    analysis()
