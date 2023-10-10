from keras import Sequential
from keras.src.layers import Dense
from tqdm import tqdm

from features import preprocess_data


def cnn():
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_names, num_classes = preprocess_data(
        '../data/video_data.csv', one_hot_encode=True)

    batch_sizes = [64, 128]
    epochs = [100, 200]
    optimizers = ['Adam']
    activations = ['relu']
    neurons = [32, 48, 64, 80, 128]
    hidden_layer2_neurons = [32, 48, 64, 80, 128]
    loss_funcs = ['categorical_crossentropy']

    param_combinations = [(bs, ep, opt, act, n, hn, lf) for bs in batch_sizes for ep in epochs for opt in optimizers for act in activations for n in neurons for hn in hidden_layer2_neurons for lf in loss_funcs]

    best_score = 0
    best_params = {}

    for bs, ep, opt, act, n, hn, lf in tqdm(param_combinations):
        model = Sequential()
        model.add(Dense(n, input_dim=X_train_scaled.shape[1], activation=act))
        model.add(Dense(hn, activation=act))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=lf, optimizer=opt, metrics=['accuracy'])

        model.fit(X_train_scaled, y_train, epochs=ep, batch_size=bs, verbose=0)

        val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)

        if val_accuracy > best_score:
            best_score = val_accuracy
            best_params = {
                'batch_size': bs,
                'epochs': ep,
                'optimizer': opt,
                'activation': act,
                'neurons': n,
                'hidden_layer2_neurons': hn,
                'loss_func': lf
            }

    print(f"Best Validation Accuracy: {best_score}")
    print(f"Best Parameters: {best_params}")