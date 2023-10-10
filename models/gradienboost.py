import xgboost as xgb
from sklearn.metrics import accuracy_score

from features import preprocess_data

OBJECTIVE = 'multi:softmax'


def gradientboost():
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_names, num_classes = preprocess_data('../data/video_data.csv')
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes)
    model.fit(X_train_scaled, y_train)

    y_preds = {'train': model.predict(X_train_scaled), 'val': model.predict(X_val_scaled), 'test': model.predict(X_test_scaled)}
    accuracies = {key: accuracy_score(y_train if key == 'train' else (y_val if key == 'val' else y_test), pred) for key, pred in y_preds.items()}
    print(f"Training: {accuracies['train']}\nValidation: {accuracies['val']}\nTest: {accuracies['test']}")