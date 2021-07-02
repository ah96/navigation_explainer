from sklearn.model_selection import train_test_split

def split_test_train(X, y, test_size_param, random_state_param):

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_param, random_state=random_state_param)

	return X_train, X_test, y_train, y_test


