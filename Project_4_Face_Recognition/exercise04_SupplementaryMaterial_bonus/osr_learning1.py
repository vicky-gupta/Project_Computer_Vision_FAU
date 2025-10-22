from typing import Tuple, Callable

import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def spl_training(x_train: np.ndarray, y_train: np.ndarray) \
        -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    
    # Create a pipeline with StandardScaler and OneVsRestClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', OneVsRestClassifier(SVC(probability=True, kernel='rbf')))
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'classifier__estimator__C': [0.1, 1, 10],
        'classifier__estimator__gamma': ['scale', 'auto']
    }
    
    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    
    # Get the best estimator
    best_estimator = grid_search.best_estimator_
    
    def spl_predict_fn(x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = best_estimator.predict(x_test)
        y_score = best_estimator.predict_proba(x_test).max(axis=1)
        return y_pred, y_score

    return spl_predict_fn



def mpl_training(x_train: np.ndarray, y_train: np.ndarray) \
        -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    
    # Create a pipeline with StandardScaler and RandomForestClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    
    # Get the best estimator
    best_estimator = grid_search.best_estimator_
    
    def mpl_predict_fn(x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = best_estimator.predict(x_test)
        y_score = best_estimator.predict_proba(x_test).max(axis=1)
        return y_pred, y_score

    return mpl_predict_fn


def load_challenge_validation_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge validation data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    # TODO: check for correct path
    path_to_challenge_validation_data = "challenge_validation_data.csv"
    df = pd.read_csv(path_to_challenge_validation_data, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y


def main():
    x_train, y_train = load_challenge_validation_data()

    # TODO: implement
    spl_predict_fn = spl_training(x_train, y_train)

    # TODO: implement
    mpl_predict_fn = mpl_training(x_train, y_train)

    # TODO: No todo, but this is roughly how we will test your implementation (with real data). So please make sure
    #       that this call (besides the unit tests) does what it is supposed to do.
    x_test = np.random.rand(50, x_train.shape[1])
    y_test = np.random.randint(-1, 5, 50)
    for predict_fn in [spl_predict_fn, mpl_predict_fn]:
        y_pred, y_score = predict_fn(x_test)
        print("Dummy acc: {}".format(np.equal(y_test, y_pred).sum() / len(x_test)))


if __name__ == '__main__':
    main()
