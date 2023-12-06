import tensorflow as tf

def create_model():
    """Create a larger and more complex model for demonstration purposes."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),

        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(1000, activation='softmax')
    ])
    return model

def generate_synthetic_data(batch_size, num_batches):
    """Generate synthetic data for training and validation."""
    for _ in range(num_batches):
        images = tf.random.normal((batch_size, 128, 128, 3))
        labels = tf.random.uniform((batch_size,), maxval=10, dtype=tf.int32)
        yield images, labels

def train_multi_gpu(batch_size, epochs):
    """Train the model using multiple GPUs."""
    # Detect and initialize GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    # Create a synthetic dataset
    train_dataset = tf.data.Dataset.from_generator(
        lambda: generate_synthetic_data(global_batch_size, 10000),
        output_signature=(
            tf.TensorSpec(shape=(global_batch_size, 128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(global_batch_size,), dtype=tf.int32)
        )
    )

    with strategy.scope():
        model = create_model()
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    model.fit(train_dataset, epochs=epochs, steps_per_epoch=10000)

if __name__ == "__main__":
    train_multi_gpu(batch_size=64, epochs=10)
