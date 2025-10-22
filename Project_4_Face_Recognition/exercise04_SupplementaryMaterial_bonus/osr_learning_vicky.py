import os
import pickle
from typing import Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# already calculated hyperparameter by testing

SPL_BEST_PARAMS = {'C': 10, 'gamma': 'scale', 'kernel': 'linear'}
MPL_BEST_PARAMS = {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}

def spl_training(x_train: np.ndarray, y_train: np.ndarray) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Note: Dummy accuracy has been ignored while choosing hyperparameter to
    to maintain the validation accuracy on test dataset in a sweet range between underfitting and overfitting.
    """
    y_spl = np.where(y_train >= 0, y_train, -1)  # Treat all KUCs as a single class
    clf = SVC(probability=True, **SPL_BEST_PARAMS)
    clf.fit(x_train, y_spl)

    def spl_predict_fn(x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = clf.predict(x_test)
        y_score = clf.predict_proba(x_test).max(axis=1)
        return y_pred, y_score

    return spl_predict_fn

def mpl_training(x_train: np.ndarray, y_train: np.ndarray) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the multi pseudo label (MPL) approach using KNN.
    """
    clf = KNeighborsClassifier(**MPL_BEST_PARAMS)
    clf.fit(x_train, y_train)

    def mpl_predict_fn(x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = clf.predict(x_test)
        y_score = clf.predict_proba(x_test).max(axis=1)
        return y_pred, y_score

    return mpl_predict_fn

def load_challenge_validation_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge validation data.
    """
    path_to_challenge_validation_data = "challenge_validation_data.csv"
    df = pd.read_csv(path_to_challenge_validation_data, header=None)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return x, y

def evaluate_model(predict_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], x_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred, _ = predict_fn(x_test)
    return accuracy_score(y_test, y_pred)

def main():
    x, y = load_challenge_validation_data()

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    spl_predict_fn = spl_training(x_train, y_train)
    mpl_predict_fn = mpl_training(x_train, y_train)

    x_dummy = np.random.rand(50, x_train.shape[1])
    y_dummy = np.random.randint(-1, 5, 50)

    for name, predict_fn in [("SPL", spl_predict_fn), ("MPL", mpl_predict_fn)]:
        dummy_accuracy = evaluate_model(predict_fn, x_dummy, y_dummy)
        print(f"Dummy acc ({name}): {dummy_accuracy:.2f}")

    # Train and test accuracy
    for name, predict_fn in [("SPL", spl_predict_fn), ("MPL", mpl_predict_fn)]:
        test_acc = evaluate_model(predict_fn, x_test, y_test)
        print(f"{name} Test accuracy: {test_acc:.2f}")

if __name__ == '__main__':
    main()