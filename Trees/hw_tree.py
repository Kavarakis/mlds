import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random


def all_columns(X, rand):
    return list(range(X.shape[1]))


def entropy_score(x, y, split_val, no_classes):
    """ 
    (np.array, np.array, number) -> number
    Calculates entropy score for given feature X, target var y and split value split_val.
    """
    cond = x <= split_val
    left = y[cond == True]
    right = y[cond == False]
    rl, rr = 0, 0
    for c in range(no_classes):
        resl = left[left == c].shape[0] / left.shape[0]
        resr = right[right == c].shape[0] / right.shape[0]
        rl -= resl * np.log2(resl)
        rr -= resr * np.log2(resr)

    score = rl * (left.shape[0] / y.shape[0]) + rr * (right.shape[0] /
                                                      y.shape[0])
    return score


def gini_score(x, y, split_val, no_classes):
    """
    (np.array, np.array, number) -> number
    Calculates gini score for given feature X, target var y and split value split_val.
    """
    cond = x <= split_val
    left = y[cond == True]
    right = y[cond == False]
    rl, rr = 0, 0
    for c in range(no_classes):
        resl = 0
        resr = 0
        try:
            resl = left[left == c].shape[0] / left.shape[0]
            resr = right[right == c].shape[0] / right.shape[0]

        except ZeroDivisionError:
            resl = resl if (resl) else 0
            resr = resr if (resr) else 0

        rl += resl**2
        rr += resr**2
    rl = 1 - rl
    rr = 1 - rr
    score = rl * (left.shape[0] / y.shape[0]) + rr * (right.shape[0] /
                                                      y.shape[0])
    return score


def get_split_candidates(x):
    x = np.unique(x)
    if x.shape[0] <= 1:
        return x
    split_candidate = []
    for index in range(x.shape[0] - 1):
        split_val = np.mean([x[index], x[index + 1]])
        split_candidate.append(split_val)
    return split_candidate


def get_score(args):
    x, y, split_val, no_classes = args
    return gini_score(x, y, split_val, no_classes)


def best_split(x, y, col_ids):
    features_score = []
    i = 0
    for val in x[:, col_ids].T:
        splits = get_split_candidates(val)
        res = [get_score([val, y, s, 2]) for s in splits]
        score = min(res)
        _id = int(res.index(score))
        features_score.append([splits[_id], score, col_ids[i], _id])
        i += 1
    features_score = np.array(features_score)
    _id = int(np.argmin(features_score[:, 1]))
    split, score, col_id = features_score[_id, :3]
    return split, score, int(col_id)


class Tree:
    def __init__(self, rand, get_candidate_columns, min_samples):
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples

    def builder(self, x, y, f_names=[], s=':', counter=0):
        counter += 1
        try:
            tree = {}
            col_ids = self.get_candidate_columns(x, self.rand)
            self.no_classes = np.unique(y).shape[0]
            if x.shape[0] < self.min_samples:
                tree['yes'] = np.count_nonzero(y)
                tree['no'] = y.shape[0] - tree['yes']
                return tree
            split_val, score, col_id = best_split(x, y, col_ids)
            cond = x[:, [col_id]].reshape(x.shape[0]) <= split_val

            left_f = x[cond == True, :]
            right_f = x[cond == False]
            try:
                if left_f.shape[0] == 0:
                    tree['yes'] = np.count_nonzero(y)
                    tree['no'] = (len(y) - tree['yes'])
                    return tree
                if right_f.shape[0] == 0:
                    tree['yes'] = np.count_nonzero(y)
                    tree['no'] = (len(y) - tree['yes'])
                    return tree
            except KeyError:
                print('Keyerror')
            left_y = y[cond == True]
            right_y = y[cond == False]

            tree['columnindex'] = col_id
            tree['splitvalue'] = split_val
            tree['condition'] = "{}<={}".format(col_id, split_val)
            if (len(f_names) > col_id):
                tree['f_name'] = f_names[col_id]
            else:
                tree['f_name'] = '-'
            tree['no'] = self.builder(left_f,
                                      left_y,
                                      f_names=f_names,
                                      s='tree[yes]',
                                      counter=counter)
            tree['yes'] = self.builder(right_f,
                                       right_y,
                                       f_names=f_names,
                                       s='tree[yes]',
                                       counter=counter)

            tree['score'] = score
            return tree
        except RecursionError:
            print('RecursionError happened ...')
            pass

    def build(self, X, y, f_names=[]):
        self.tree = self.builder(X, y, f_names)
        return self

    def predict(self, features):
        pred = []
        for row in features:
            p = self.get_predict(row, self.tree)
            pred.append(p)
        return pred

    def get_predict(self, feature, tree):

        if 'splitvalue' not in tree.keys():
            if (tree['yes'] > tree['no']):
                return 1
            elif (tree['yes'] < tree['no']):
                return 0
            else:
                return np.random.choice([0, 1])
        col_id = tree['columnindex']
        val = tree['splitvalue']
        if (feature[col_id] <= val):
            result = self.get_predict(feature, tree['no'])
        else:
            result = self.get_predict(feature, tree['yes'])

        return result

    def get_predict_proba(self, feature, tree):

        if 'splitvalue' not in tree.keys():
            if (tree['yes'] > tree['no']):
                res = tree['yes']/(tree['yes']+tree['no'])
                return [1-res, res]
            elif (tree['yes'] < tree['no']):
                res = tree['no']/(tree['yes']+tree['no'])
                return [res, 1-res]
            else:
                return [0.5, 0.5]
        col_id = tree['columnindex']
        val = tree['splitvalue']
        if(feature[col_id] <= val):
            result = self.get_predict_proba(feature, tree['no'])
        else:
            result = self.get_predict_proba(feature, tree['yes'])

        return result

    def predict_proba(self, features):

        pred = []
        for row in features:
            p = self.get_predict_proba(row, self.tree)
            pred.append(p)
        return np.array(pred).reshape((features.shape[0], 2))


def get_candidate_columns(features, rand):
    return list(range(len(features[0])))


def random_sqrt_columns(X, rand):
    return [
        rand.choice(list(range(X.shape[1])))
        for i in range(np.round(np.sqrt(X.shape[1])))
    ]


# HELPER FUNCTIONS
def get_candidate_columns_default(features, rand):
    return list(range(len(features[0])))


class RandomForest():
    def __init__(self, rand, n, min_samples=2):

        self.min_samples = min_samples
        self.rand = rand
        self.n = n
        self.trees = []

    def rf_get_candidate_columns(self, x, rand):
        res = sorted(
            rand.sample(
                list(range(x.shape[1])),
                rand.choice(np.arange(int(x.shape[1] / 2), x.shape[1]))))
        return res

    def bootstrap(self, x, y):

        data = np.append(x, y.reshape(y.shape[0], 1), axis=1)
        data = np.array([self.rand.choice(data) for x in range(data.shape[0])])
        return data[:, :-1], data[:, -1]

    def build(self, x, y):
        self.x = x
        self.y = y
        for i in range(self.n):
            x, y = self.bootstrap(x, y)
            instance = Tree(self.rand, get_candidate_columns_default,
                            self.min_samples)
            self.trees.append(instance.build(x, y))
        return self

    def predict(self, x):
        res = []
        for tree in self.trees:
            res.append(tree.predict(x))
        res = np.array(res)
        return np.round(np.mean(res, axis=0))

    def importance(self):
        baseline = np.sum(self.predict(self.x) != self.y) / len(self.y)
        res = []
        for i in range(self.x.shape[1]):
            temp = np.array(self.x, copy=True)
            temp[:, i] = self.rand.sample(temp[:, i].tolist(), temp.shape[0])
            mcr = np.sum(self.predict(temp) != self.y) / len(self.y)

            res.append(mcr - baseline)
        return res


def get_bootstrap(x, y, rand):
    data = np.append(x, y.reshape(y.shape[0], 1), axis=1)
    data = np.array([rand.choice(data) for x in range(data.shape[0])])
    return data[:, :-1], data[:, -1]


def eval_model(x_train, y_train, x_test, y_test, model, n_boot, rand):
    res = {'t': [], 'p': [], 'tt': [], 'tp': []}
    for i in range(n_boot):
        x_train, y_train = get_bootstrap(x_train, y_train, rand)
        x_test, y_test = get_bootstrap(x_test, y_test, rand)
        model.build(x_train, y_train)
        res['p'].append(model.predict(x_test))
        res['t'].append(y_test)
        res['tp'].append(model.predict(x_train))
        res['tt'].append(y_train)
    return res


def hw_tree_full(train, test):
    x_train, y_train = train
    x_test, y_test = test
    n_boot = 10
    rand = random.Random(1)
    model = Tree(rand, get_candidate_columns_default, 2)
    res = eval_model(x_train, y_train, x_test, y_test, model, n_boot, rand)
    misclass_res = []
    err_res = []

    def calc(yp, yt):
        yt = yt.astype(int)
        mcr = 1. - (np.sum(yp == yt) / len(yp))
        misclass_res.append(mcr)
        err = np.sum(yp != yt) / len(yp)
        err_res.append(err)

    # test
    for yp, yt in zip(res['p'], res['t']):
        calc(yp, yt)
    test = (np.mean(misclass_res),
            np.std(err_res) / np.sqrt(len(misclass_res)))
    misclass_res = []
    err_res = []
    # train
    for yp, yt in zip(res['tp'], res['tt']):
        calc(yp, yt)
    train = (np.mean(misclass_res),
             np.std(err_res) / np.sqrt(len(misclass_res)))
    return train, test


def hw_randomforests(train, test):
    x_train, y_train = train
    x_test, y_test = test
    n_boot = 10
    rand = random.Random(1)
    model = RandomForest(rand, 100)
    res = eval_model(x_train, y_train, x_test, y_test, model, n_boot, rand)
    misclass_res = []
    err_res = []

    def calc(yp, yt):
        yt = yt.astype(int)
        mcr = 1. - (np.sum(yp == yt) / len(yp))
        misclass_res.append(mcr)
        err = np.sum(yp != yt) / len(yp)
        err_res.append(err)

    # test
    for yp, yt in zip(res['p'], res['t']):
        calc(yp, yt)
    test = (np.mean(misclass_res),
            np.std(err_res) / np.sqrt(len(misclass_res)))
    misclass_res = []
    err_res = []
    # train
    for yp, yt in zip(res['tp'], res['tt']):
        yt = yt.astype(int)
        mcr = 1. - (np.sum(yp == yt) / len(yp))
        misclass_res.append(mcr)
        err = np.sum(yp != yt) / len(yp)
        err_res.append(err)
    train = (np.mean(misclass_res),
             np.std(err_res) / np.sqrt(len(misclass_res)))
    return train, test


def plot_rf(train, test):
    n_boot = 100
    x_train, y_train = train
    x_test, y_test = test
    rand = random.Random(1)
    n = 30
    no_trees = list(range(1, n + 1))

    def calc(yp, yt):
        yt = yt.astype(int)
        mcr = 1. - (np.sum(yp == yt) / len(yp))
        misclass_res.append(mcr)
        err = np.sum(yp != yt) / len(yp)
        err_res.append(err)

    testr = []
    trainr = []
    rr = []
    se_train = []
    se_test = []
    plt.rc('font', size=10)
    plt.rcParams['font.size'] = 11
    fig, a = plt.subplots(2, 2, sharex=True)
    fig.set_size_inches(20, 10)
    fig.set_dpi(100)
    plt.rcParams['font.size'] = 11

    plt.grid()
    for i in no_trees:
        model = RandomForest(rand, i)
        res = eval_model(x_train, y_train, x_test, y_test, model, n_boot, rand)
        # test
        misclass_res = []
        err_res = []
        for yp, yt in zip(res['p'], res['t']):
            calc(yp, yt)
        testr.append(misclass_res)
        # train
        misclass_res = []
        err_res = []
        for yp, yt in zip(res['tp'], res['tt']):
            calc(yp, yt)
        trainr.append(misclass_res)
        rr.append(np.mean(testr))
        se_test.append(np.std(testr) / np.sqrt(len(testr)))
        se_train.append(np.std(trainr) / np.sqrt(len(trainr)))

        a[0][0].boxplot(testr, showmeans=True)
        a[0][1].boxplot(trainr, showmeans=True)

        print('Finished with rf for n =', i)
    a[0][0].set_title('Test data', fontsize=11)
    a[0][1].set_title('Train data', fontsize=11)
    a[0][0].set_ylabel('Misclassification rate', fontsize=11)
    a[1][0].plot(no_trees, se_test)
    a[1][1].plot(no_trees, se_train)

    a[1][0].set_title('Misclassification rate SE - test data', fontsize=11)
    a[1][1].set_title('Misclassification rate SE - train data', fontsize=11)
    a[1][0].set_ylabel('SE value', fontsize=11)

    for x in a.ravel():
        x.set_xticks(no_trees)
        x.set_xticklabels(tuple(no_trees))
        x.set_xlabel('n', fontsize=11)
        x.grid()

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.rcParams.update({'font.size': 11})
    plt.grid()
    fig.tight_layout()
    fig.savefig('rf_mcr.png')
    plt.show()


def get_mcr_plot(model, rand, x, y, xt, yt, name):
    n_boot = 100
    res_train = []
    res_test = []
    for i in range(n_boot):
        x_train, y_train = get_bootstrap(x, y, rand)
        x_test, y_test = get_bootstrap(xt, yt, rand)
        model.build(x_train, y_train)
        res_test.append(np.sum(model.predict(xt) != yt) / len(yt))
        res_train.append(np.sum(model.predict(x) != y) / len(y))
    fig, a = plt.subplots(1, 2)

    plt.gca().set_title(f'Misclassification rates - {name}')
    a[0].boxplot(res_train, showmeans=True)
    a[0].set_title('Train data')
    a[1].boxplot(res_test, showmeans=True)
    a[1].set_title('Test data')
    a[0].grid()

    a[0].set_xticklabels([])
    a[0].set_xticks([])
    a[1].set_xticklabels([])
    a[1].set_xticks([])
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'mcr-{name}.png')
    plt.show()


def get_importance(rand, X, y, labels):
    clf = RandomForest(rand, n=100, min_samples=5)
    clf.build(X, y)
    cols = []
    for t in clf.trees:
        if 'columnindex' in t.tree:
            cols.append(t.tree['columnindex'])

    imp = clf.importance()
    _cols, freq = np.unique(cols, return_counts=True)
    plt.grid()
    plt.rcParams['font.size'] = 11
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.scatter(_cols, freq / len(cols), label='RF root feature')
    plt.scatter(_cols, [imp[i] for i in _cols],
                marker='^',
                s=100,
                label='Variable importance increase',
                alpha=0.4)
    plt.xticks(_cols, labels=_cols)
    plt.legend(loc='best')
    plt.ylabel('Percents (%)')
    plt.xlabel('Variables')
    plt.title('Measure of variable importance')
    plt.savefig('importance.png')
    plt.show()


def main():
    rand = random.Random(1)
    _df = pd.read_csv('tki-resistance.csv')
    _df['Class'] = _df['Class'].astype('category').cat.codes
    df = _df.loc[:130]
    df_test = _df.loc[130:]
    X = df[df.columns[df.columns != 'Class'].values].values
    y = df['Class'].values
    X_test = df_test[df_test.columns[df_test.columns != 'Class'].values].values
    y_test = df_test['Class'].values

    print(
        'Tree - Train(misclassification rate, SE),Test(misclassification rate, SE):'
    )
    print(hw_tree_full((X, y), (X_test, y_test)))
    print(
        'RF - Train(misclassification rate, SE),Test(misclassification rate, SE):'
    )
    print(hw_randomforests((X, y), (X_test, y_test)))
    plot_rf((X, y), (X_test, y_test))
    print('get_mcr plots...')
    get_mcr_plot(Tree(rand, all_columns, 2), rand, X, y, X_test, y_test,
                 'Tree')
    get_mcr_plot(RandomForest(rand, 100, 2), rand, X, y, X_test, y_test, 'RF')
    print('get_importance plots...')

    get_importance(rand, X, y, list(range(X.shape[1])))


if __name__ == "__main__":
    main()
