import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def get_cifar100_datasets(image_size=(96, 96), batch_size=64, val_split=0.1):
    # Load CIFAR-100 data
    (X_train_full, y_train_full), (X_test, y_test) = cifar100.load_data(label_mode='fine')

    # One-hot encode the labels
    enc = OneHotEncoder(sparse_output=False)
    y_train_full = enc.fit_transform(y_train_full)
    y_test = enc.transform(y_test)

    # Split training into train + val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_split,
        random_state=42,
        stratify=y_train_full
    )

    # Preprocessing function
    def preprocess(image, label):
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def build_dataset(X, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_dataset = build_dataset(X_train, y_train, shuffle=True)
    val_dataset = build_dataset(X_val, y_val)
    test_dataset = build_dataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset
