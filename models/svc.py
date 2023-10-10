import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from features import preprocess_data

KERNEL = 'linear'
C_VALUE = 1


def svc():
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_names, num_classes = preprocess_data(
        '../data/video_data.csv')

    svc_model = SVC(kernel=KERNEL, C=C_VALUE)
    svc_model.fit(X_train_scaled, y_train)

    y_val_pred = svc_model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    y_test_pred = svc_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f'Validation Accuracy: {val_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')

    plt.figure(figsize=(10, 8))
    importance = svc_model.coef_[0]
    plt.barh(range(len(importance)), importance)
    plt.yticks(range(len(importance)), feature_names)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('../img/svc_features.png', dpi=300)
    plt.show()