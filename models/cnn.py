from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

from features import preprocess_data

EPOCHS = 65
BATCH_SIZE = 16
HIDDEN_LAYERS = [(64, 'relu'), (32, 'relu')]
OPTIMIZER = 'adam'
LOSS_FUNC = 'categorical_crossentropy'

def cnn():
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, _, num_classes = preprocess_data(
        '../data/video_data.csv', one_hot_encode=True)

    model = build_model(X_train_scaled.shape[1], num_classes)
    model.fit(X_train_scaled, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val_scaled, y_val))

    evaluate_model(model, X_val_scaled, y_val, X_test_scaled, y_test)

def build_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(HIDDEN_LAYERS[0][0], input_dim=input_dim, activation=HIDDEN_LAYERS[0][1]))
    for neurons, activation in HIDDEN_LAYERS[1:]:
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=LOSS_FUNC, optimizer=OPTIMIZER, metrics=['accuracy'])
    return model

def evaluate_model(model, X_val, y_val, X_test, y_test):
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Validation Accuracy: {val_accuracy}\nTest Accuracy: {test_accuracy}')
    plot_model(model, to_file='../img/model_plot.png', show_shapes=True, show_layer_names=True)