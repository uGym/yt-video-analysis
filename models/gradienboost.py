import xgboost as xgb
from sklearn.metrics import accuracy_score

from features import preprocess_data

OBJECTIVE = 'multi:softmax'


def gradientboost():
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_names, num_classes = preprocess_data('../data/video_data.csv', one_hot_encode=True)
    model = xgb.XGBClassifier(objective=OBJECTIVE, num_class=num_classes)
    model.fit(X_train_scaled, y_train)

    y_val_pred = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f'Validation Accuracy: {val_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')