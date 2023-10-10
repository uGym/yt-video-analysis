import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

from features import preprocess_data

MAX_DEPTH = 14
MIN_SAMPLES_LEAF = 1
MIN_SAMPLES_SPLIT = 6
N_ESTIMATORS = 50
CRITERION = 'gini'

def randomforest():
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_names, _ = preprocess_data('../data/video_data.csv')
    rf_model = RandomForestClassifier(criterion=CRITERION, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, min_samples_split=MIN_SAMPLES_SPLIT, n_estimators=N_ESTIMATORS)
    rf_model.fit(X_train_scaled, y_train)

    y_preds = { 'train': rf_model.predict(X_train_scaled), 'val': rf_model.predict(X_val_scaled), 'test': rf_model.predict(X_test_scaled) }
    accuracies = { key: accuracy_score(y_train if key == 'train' else (y_val if key == 'val' else y_test), pred) for key, pred in y_preds.items() }
    print(f"Training Accuracy: {accuracies['train']}\nValidation Accuracy: {accuracies['val']}\nTest Accuracy: {accuracies['test']}")

    plot_features_and_tree(rf_model, feature_names)


def plot_features_and_tree(model, feature_names):
    importances = model.feature_importances_
    sorted_indices = importances.argsort()[::-1]
    single_tree = model.estimators_[0]

    fig, axes = plt.subplots(1, 2, figsize=(30, 10))

    plot_tree(single_tree, filled=True, feature_names=feature_names, class_names=['low', 'medium', 'high'], rounded=True, ax=axes[0])
    axes[0].set_title('Decision Tree')

    axes[1].barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
    axes[1].set_yticks(range(len(sorted_indices)))
    axes[1].set_yticklabels([feature_names[i] for i in sorted_indices])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Feature Importance')
    axes[1].set_title('Feature Importances')

    plt.tight_layout()
    plt.savefig('../img/combined_plots.png', dpi=300)
    plt.show()