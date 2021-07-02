import numpy as np


def train(X_train, y_train, mode):
    if mode == 'regression':

        from sklearn import datasets, linear_model

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(np.array(X_train), np.array(y_train))

        return regr

    elif mode == 'classification':

        from sklearn import tree

        clf = tree.DecisionTreeClassifier()

        clf = clf.fit(np.array(X_train), np.array(y_train))

        return clf

    elif mode == 'random_forest':

        import sklearn.ensemble

        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)

        rf.fit(np.array(X_train), np.array(y_train))

        return rf
