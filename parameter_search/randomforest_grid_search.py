from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from features import preprocess_data


def randomforest_grid_search():

    best_params = None
    best_val_accuracy = 0

    alphas = [0.015, 0.02]
    betas = [1.035, 1.04]
    gammas = [0.71, 0.72]
    deltas = [0.98, 1.0]
    max_depths = [14, 15]
    min_samples_leafs = [1, 2]
    min_samples_splits = [6, 7]
    n_estimators = [45, 50]
    lower_qs = [0.18, 0.2]

    param_grid = product(alphas, betas, gammas, deltas, max_depths, min_samples_leafs, min_samples_splits, n_estimators, lower_qs)
    total_combinations = len(list(product(alphas, betas, gammas, deltas, max_depths, min_samples_leafs, min_samples_splits, n_estimators, lower_qs)))

    for alpha, beta, gamma, delta, max_depth, min_samples_leaf, min_samples_split, n_estimator, lower_q in tqdm(param_grid, total=total_combinations):
        X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_names = preprocess_data(
            '../data/video_data.csv')

        rf_model = RandomForestClassifier(criterion='gini', max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                          min_samples_split=min_samples_split, n_estimators=n_estimator)
        rf_model.fit(X_train_scaled, y_train)

        y_val_pred = rf_model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta, 'max_depth': max_depth,
                           'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
                           'n_estimator': n_estimator, 'lower_q': lower_q}

    print(f"Best Parameters: {best_params}, Best Validation Accuracy: {best_val_accuracy}")