import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

TRAIN_DIR = 'dataset/Train'
TEST_DIR = 'dataset/Test'
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 123
EPOCHS = 3
NUM_CLASSES = 43

def load_datasets():

    train_ds = image_dataset_from_directory(
        directory=TRAIN_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        subset="training",
        seed=SEED,
        validation_split=VAL_SPLIT,
        image_size=IMG_SIZE
    )

    val_ds = image_dataset_from_directory(
        directory=TRAIN_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        subset="validation",
        seed=SEED,
        validation_split=VAL_SPLIT,
        image_size=IMG_SIZE
    )

    test_ds = image_dataset_from_directory(
        directory=TEST_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    return train_ds, val_ds, test_ds

def build_model(input_shape, num_classes):

    model = models.Sequential([

        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Dropout(0.2),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_ds, val_ds, epochs):

    callbacks = [
        callbacks.TensorBoard(log_dir='logs'),
        callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    return history

def evaluate_model(model, test_ds):

    test_loss, test_acc = model.evaluate(test_ds)
    print(f'\nTočnost na testnom skupu: {test_acc:.4f}')
    return test_acc

def plot_confusion_matrix(y_true, y_pred, class_labels):

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.title('Matrica zabune za testni skup')
    plt.xlabel('Predviđene klase')
    plt.ylabel('Stvarne klase')
    plt.tight_layout()
    plt.show()

def main():

    train_ds, val_ds, test_ds = load_datasets()

    y_test = tf.concat([y for (x, y) in test_ds], axis=0)
    y_test = tf.argmax(y_test, axis=1)

    model = build_model((48, 48, 3), NUM_CLASSES)
    model.summary()

    print("\nPočetak treniranja...")
    train_model(model, train_ds, val_ds, EPOCHS)

    best_model = models.load_model('best_model.h5')
    test_acc = evaluate_model(best_model, test_ds)

    y_test_pred = best_model.predict(test_ds)
    y_test_pred_classes = tf.argmax(y_test_pred, axis=1)
    
    plot_confusion_matrix(y_test, y_test_pred_classes, np.arange(NUM_CLASSES))

if __name__ == "__main__":
    main()
