import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b, fmin_slsqp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import seaborn as sns
import random

MAXFUN = 1000
MAXITER = 10000
MBOG_TRAIN = 10000


def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res


class MultinomialLogReg:
    def __init__(self, f=True):
        self.w = np.array([])
        self.c = np.array([])
        self.L = 0
        self.K = 0
        self.mle_val = []
        self.f = False

    def neg_log_likelihood_multinomial(self, beta):
        beta = np.reshape(beta,
                          newshape=(np.shape(self.X)[1] + 1, len(self.c) - 1))
        value = 0

        for i in range(0, len(self.X)):
            firstsum = 0
            secondsum = 0
            for j in range(0, len(self.c) - 1):
                beta_coeffs = beta[:, j]
                dot = np.dot(self.x_int[i], beta_coeffs)
                firstsum += self.y_mat[i][j] * dot
                secondsum += np.exp(dot)
            value = value + np.log(1 + secondsum) - firstsum
        self.mle_val.append(value)
        return value

    def build(self, X, y):
        self.mle_val = []
        self.X = X
        self.y = y
        self.c = np.unique(y)
        self.L = self.c.shape[0]
        self.y_mat = np.zeros(shape=(len(X), self.c.shape[0]))
        for i, y_instance in enumerate(y):
            ind = np.where(self.c == y_instance)
            self.y_mat[i][ind[0]] = 1

        scaler = StandardScaler()
        scaler.fit(X).transform(X, copy=False)
        self.K = np.shape(X)[1]
        w_init = np.zeros(shape=(np.shape(X)[1] + 1, len(self.c) - 1))
        self.x_int = np.column_stack((np.ones(shape=(len(X), 1)), X))
        func = fmin_l_bfgs_b if self.f == True else fmin_slsqp

        self.weights, value, d = fmin_l_bfgs_b(
            self.neg_log_likelihood_multinomial,
            w_init,
            approx_grad=True,
            epsilon=1e-8,
            maxiter=MAXITER,
            pgtol=1e-5,
            maxfun=MAXFUN)
        print("Value of function after optimization: ", value)
        self.w = self.weights
        self.weights = np.reshape(self.weights,
                                  newshape=(self.K + 1, self.L - 1))
        return self

    def predict(self, X):
        scaler = StandardScaler()
        scaler.fit(X).transform(X, copy=False)
        x_int = np.column_stack((np.ones(shape=(len(X), 1)), X))
        res = np.matmul(x_int, self.weights)
        probabilities = np.exp(res)
        predictions = np.array([])
        final_probabilities = np.empty(shape=(probabilities.shape[0],
                                              probabilities.shape[1] + 1))
        for j, row in enumerate(probabilities):
            old = row
            row = np.append(row, 1 / (1 + np.sum(row)))
            for i in range(0, len(row) - 1):
                temp = row[i] * row[len(row) - 1]
                if (np.isnan(temp)):
                    temp = 0
                row[i] = row[i] * row[len(row) - 1]
            final_probabilities[j] = row
            index = np.argmax(row)
            prediction = self.c[index]
            predictions = np.append(predictions, prediction)

        return final_probabilities


class OrdinalLogReg:
    def __init__(self):
        self.weights = np.array([])
        self.classes = np.array([])
        self.beta_length = 0
        self.threshold_length = 0
        self.mle_val = []

    def build(self, X, y):
        self.mle_val = []
        scaler = StandardScaler()
        scaler.fit(X).transform(X, copy=False)

        self.classes = np.array(list(set(y)))
        self.threshold_length = len(self.classes) - 2
        self.beta_length = np.shape(X)[1] + 1
        weights_init = np.ones(shape=(self.beta_length + self.threshold_length,
                                      1))
        bounds = np.empty(shape=(self.beta_length + self.threshold_length, 2))

        for i in range(0, self.beta_length):
            bounds[i][0] = -math.inf
            bounds[i][1] = math.inf

        for i in range(self.beta_length,
                       self.beta_length + self.threshold_length):
            bounds[i][0] = 1e-5
            bounds[i][1] = math.inf

        def neg_log_likelihood_ordinal(weights):
            beta = weights[:self.beta_length]
            thresholds = np.concatenate(
                ([-math.inf,
                  0], np.cumsum(weights[self.beta_length:]), [math.inf]))

            value = 0
            for i in range(0, len(X)):
                for j, k in enumerate(self.classes):
                    if y[i] == k:
                        value = value - np.log(
                            sigmoid(thresholds[j + 1] -
                                    np.dot(beta[1:], X[i]) - beta[0]) -
                            sigmoid(thresholds[j] - np.dot(beta[1:], X[i]) -
                                    beta[0]))
            self.mle_val.append(value)
            return value

        self.weights, value, d = fmin_l_bfgs_b(neg_log_likelihood_ordinal,
                                               x0=weights_init,
                                               approx_grad=True,
                                               epsilon=1e-8,
                                               maxiter=MAXITER,
                                               pgtol=1e-5,
                                               maxfun=MAXFUN,
                                               bounds=bounds)
        print("Value of function after optimization: ", value)
        self.w = self.weights
        return self

    def predict(self, X):
        beta = self.weights[:self.beta_length]
        thresholds = np.concatenate(
            ([-math.inf,
              0], np.cumsum(self.weights[self.beta_length:]), [math.inf]))
        scaler = StandardScaler()
        scaler.fit(X).transform(X, copy=False)

        probabilities = [[
            sigmoid(thresholds[k] - np.dot(beta[1:], example) - beta[0]) -
            sigmoid(thresholds[k - 1] - np.dot(beta[1:], example) - beta[0])
            for k in range(1, len(thresholds))
        ] for example in X]

        predictions = np.array([])
        for row in probabilities:
            index = np.argmax(row)
            prediction = self.classes[index]
            predictions = np.append(predictions, prediction)

        return np.array(probabilities)


def get_cross_validation_score(X, y, k, model, classes):
    data = np.column_stack((X, y))
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

        y_matrix_test = np.zeros(shape=(len(X_test), len(classes)))
        for i, y_instance in enumerate(y_test):
            ind = np.where(classes == y_instance)
            y_matrix_test[i][ind[0]] = 1

        y_matrix_train = np.zeros(shape=(len(X_train), len(classes)))
        for i, y_instance in enumerate(y_train):
            ind = np.where(classes == y_instance)
            y_matrix_train[i][ind[0]] = 1

        t = model.build(X_train, y_train)
        probabilities_test = t.predict(X_test)
        probabilities_train = t.predict(X_train)
        predictions_test = np.apply_along_axis(lambda x: np.argmax(x) + 1, 1,
                                               probabilities_test)
        predictions_train = np.apply_along_axis(lambda x: np.argmax(x) + 1, 1,
                                                probabilities_train)

        scores.append([
            log_loss(y_matrix_train, probabilities_train),
            log_loss(y_matrix_test, probabilities_test)
        ])

    train_score = np.mean([x[0] for x in scores])
    test_score = np.mean([x[1] for x in scores])

    train_uncertainty = np.std([x[0] for x in scores])
    test_uncertainty = np.std([x[1] for x in scores])
    print('test_acc: ', np.sum(y_test == predictions_test) / len(y_test))
    print('train_acc: ', np.sum(y_train == predictions_train) / len(y_train))

    return train_score, test_score, train_uncertainty, test_uncertainty


def get_bootstrap(data, size):
    rep = []
    for i in range(size):
        sample = np.random.choice(range(data.shape[0]), size=len(data))
        rep.append((data[sample, :-1], np.array(data[sample, -1], dtype=int)))
    return rep


def analyze(data, n_boot, model, label):
    rep = get_bootstrap(data, n_boot)
    coeff = []
    mle_df = pd.DataFrame()
    scores = []
    mle = []
    ii = 0
    for X, y in rep:
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.33,
                                                            random_state=42)
        t = model.build(X_train, y_train)
        coeff.append(t.w)
        mle.append(t.mle_val)
        classes = np.unique(y)

        y_matrix_test = np.zeros(shape=(len(X_test), len(classes)))
        for i, y_instance in enumerate(y_test):
            ind = np.where(classes == y_instance)
            y_matrix_test[i][ind[0]] = 1

        y_matrix_train = np.zeros(shape=(len(X_train), len(classes)))
        for i, y_instance in enumerate(y_train):
            ind = np.where(classes == y_instance)
            y_matrix_train[i][ind[0]] = 1

        probabilities_test = t.predict(X_test)
        probabilities_train = t.predict(X_train)
        predictions_test = np.apply_along_axis(lambda x: np.argmax(x) + 1, 1,
                                               probabilities_test)
        predictions_train = np.apply_along_axis(lambda x: np.argmax(x) + 1, 1,
                                                probabilities_train)
        try:

            scores.append([
                log_loss(y_matrix_train, probabilities_train),
                log_loss(y_matrix_test, probabilities_test)
            ])
        except ValueError:
            print("!")

        print('test_acc: ', np.sum(y_test == predictions_test) / len(y_test))
        print('train_acc: ',
              np.sum(y_train == predictions_train) / len(y_train))

        mle_df = pd.concat(
            [mle_df, pd.DataFrame({
                'n_boot': ii,
                'Value': t.mle_val
            })])
        ii += 1
    train_score = np.mean([x[0] for x in scores])
    test_score = np.mean([x[1] for x in scores])
    train_uncertainty = np.std([x[0] for x in scores])
    test_uncertainty = np.std([x[1] for x in scores])
    print('TEST:', test_score, test_uncertainty)
    print('TRAIN:', train_score, train_uncertainty)
    temp = pd.DataFrame(scores, columns=['train', 'test'])
    sns.boxplot(data=temp)
    plt.title('Performance on the dataset - ' + label)
    plt.savefig(f'ds-{label}.png')
    plt.grid()
    # plt.show()
    plt.clf()
    return mle_df, coeff


def multinomial_bad_ordinal_good(n, rand):
    x = np.array([rand.gauss(185, 5) for i in range(n)])
    y = 30 + np.array([rand.betavariate(4, 4) for i in range(n)]) * 150
    z = np.array([rand.choice([0, 1]) for i in range(n)])
    df = pd.DataFrame({'height': x, 'weight': y, 'sex': z})
    cat = ['slim', 'perfect', 'obese']
    criteria_perfect_m = df[(df['height'] - 100 > df['weight'] - 10)
                            & (df['height'] - 100 < df['weight'] + 10) &
                            (df['sex'] == 0)].index
    criteria_perfect_f = df[(df['height'] - 120 > df['weight'] - 10)
                            & (df['height'] - 120 < df['weight'] + 10) &
                            (df['sex'] == 1)].index

    criteria_slim_m = df[(df['height'] - 100 >= df['weight'] - 10)
                         & (df['sex'] == 0)].index
    criteria_slim_f = df[(df['height'] - 120 >= df['weight'] - 10)
                         & (df['sex'] == 1)].index

    criteria_obese_m = df[(df['height'] - 100 <= df['weight'] + 10)
                          & (df['sex'] == 0)].index
    criteria_obese_f = df[(df['height'] - 120 <= df['weight'] + 10)
                          & (df['sex'] == 1)].index

    df.loc[criteria_slim_f, 'y'] = 0
    df.loc[criteria_slim_m, 'y'] = 0

    df.loc[criteria_obese_f, 'y'] = 2
    df.loc[criteria_obese_m, 'y'] = 2

    df.loc[criteria_perfect_f, 'y'] = 1
    df.loc[criteria_perfect_m, 'y'] = 1
    X = df[df.columns[:-1]].to_numpy()
    y = df['y'].to_numpy()
    return X, y, df


def get_coeff(c):
    means = np.apply_along_axis(lambda x: np.mean(x), 0, c)
    std = np.apply_along_axis(lambda x: np.std(x), 0, c)
    temp = pd.DataFrame()
    col = ['Intercept'] + df.iloc[:, :-1].columns.tolist()
    arr = np.split(means, 5)
    arr_std = np.split(std, 5)
    for i, a in enumerate(arr):
        tt = pd.DataFrame({
            'VariableName': col,
            'Mean': a,
            'Std': arr_std[i],
            'Categories': np.full(len(a), i)
        })
        temp = pd.concat([temp, tt])
    sns.barplot(data=temp, y='VariableName', x='Mean', hue='Categories')
    plt.title('Multinomial Log. Reg. Coefficients (Means)(n_boot=5)')
    plt.savefig('coeff_mean.png')
    # plt.show()
    plt.clf()
    sns.barplot(data=temp, y='VariableName', x='Std', hue='Categories')
    plt.title('Multinomial Log. Reg. Coefficients (SD)(n_boot=5)')
    plt.savefig('coeff_std.png')
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    df = pd.read_csv("dataset_encoded.csv")
    # df = df.sample(1000)
    np.random.seed(0)
    y = df['target'].to_numpy()
    X = df.iloc[:, 1:].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=42)
    classes = np.unique(y)
    mlr, coeff1 = analyze(df.to_numpy(), 10, MultinomialLogReg(),
                          'MultinomialLogReg')
    print('Finished with analyze of MLR')
    olr, coeff2 = analyze(df.to_numpy(), 10, OrdinalLogReg(), 'OrdinalLogReg')
    print('Finished with analyze of OLR')

    plt.grid()
    plt.rcParams["figure.figsize"] = (16, 8)
    sns.lineplot(x=mlr.index.tolist(),
                 y=mlr['Value'].tolist(),
                 label='Multinomial')
    sns.lineplot(x=olr.index.tolist(),
                 y=olr['Value'].tolist(),
                 label='Ordinal')
    plt.legend(loc='best')
    plt.title('MLE performance on the dataset')
    plt.xlabel('Steps')
    plt.ylabel('Function Value')
    plt.savefig('mle-both.png')
    print('Plotted MLE for both!')
    # plt.show()
    plt.clf()
    get_coeff(coeff1)
    print('Finished COEFF!')

    print('Starting DGP...')
    X, y, dgp = multinomial_bad_ordinal_good(MBOG_TRAIN, random.Random(10))
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=42)
    print('ORDINAL:')
    clf1 = OrdinalLogReg().build(X_train, y_train)
    print('MULTINOMIAL:')

    clf2 = MultinomialLogReg().build(X_train, y_train)
    print('---------------\n')
    pred1 = clf1.predict(X_test)
    pred1 = np.apply_along_axis(lambda x: np.argmax(x), 1, pred1)
    print(np.sum((pred1 == y_test)) / len(y_test))
    pred2 = clf2.predict(X_test)
    pred2 = np.apply_along_axis(lambda x: np.argmax(x), 1, pred2)
    print(np.sum((pred2 == y_test)) / len(y_test))
    plt.rcParams["figure.figsize"] = (16, 8)
    plt.grid(b=True, which='major')
    plt.plot(range(len(clf1.mle_val)), clf1.mle_val, label='Ordinal')
    plt.plot(range(len(clf2.mle_val)), clf2.mle_val, label='Multinomial')
    plt.xlabel('Iteration steps')
    plt.ylabel('Function Value')
    plt.legend(loc='best')
    plt.title('MLE for DGP')
    plt.tight_layout()
    plt.savefig('mle_dgp.png')
    # plt.show()
    plt.clf()
    print('Ordinal:', clf1.mle_val[-1])
    print('Multinomial:', clf2.mle_val[-1])
