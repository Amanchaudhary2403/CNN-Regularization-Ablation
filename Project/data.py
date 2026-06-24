import keras
import Config
import numpy as np

def load_data():
    (x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.cifar10.load_data()

    # Normalize
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test_full = x_test_full.astype("float32") / 255.0

    # Flatten labels
    y_train_full = y_train_full.flatten()
    y_test_full = y_test_full.flatten()

    np.random.seed(Config.seed)

    # Train + Val from training set
    idx = np.random.permutation(len(x_train_full))
    total = Config.train_size + Config.val_size
    x_train_val = x_train_full[idx[:total]]
    y_train_val = y_train_full[idx[:total]]

    x_train = x_train_val[:Config.train_size]
    y_train = y_train_val[:Config.train_size]
    x_val = x_train_val[Config.train_size:]
    y_val = y_train_val[Config.train_size:]

    # Test from official test set
    idx_test = np.random.permutation(len(x_test_full))
    x_test = x_test_full[idx_test[:Config.test_size]]
    y_test = y_test_full[idx_test[:Config.test_size]]

    # One-hot encode labels for categorical_crossentropy
    y_train = keras.utils.to_categorical(y_train, Config.num_classes)
    y_val = keras.utils.to_categorical(y_val, Config.num_classes)
    y_test = keras.utils.to_categorical(y_test, Config.num_classes)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)