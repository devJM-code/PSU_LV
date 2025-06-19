import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    return (x_train, y_train), (x_test, y_test)

def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model(model, x_train, y_train):
    callbacks = [
        TensorBoard(log_dir='./logs', histogram_freq=1),
        ModelCheckpoint('best_model.h5', 
                       monitor='val_accuracy',
                       save_best_only=True,
                       mode='max',
                       verbose=1)
    ]
    
    history = model.fit(x_train, y_train,
                       epochs=10,
                       batch_size=64,
                       validation_split=0.1,
                       callbacks=callbacks)
    
    return history

def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTestna točnost: {test_acc:.4f}")
    return test_acc

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title(title)
    plt.xlabel('Predviđene klase')
    plt.ylabel('Stvarne klase')
    plt.show()

def main():
 
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = build_cnn_model()
    model.summary()

    print("\nPočetak treniranja modela...")
    train_model(model, x_train, y_train)
 
    best_model = tf.keras.models.load_model('best_model.h5')
  
    print("\nEvaluacija najboljeg modela:")
    _, train_acc = best_model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"Točnost na trening skupu: {train_acc:.4f}")
    print(f"Točnost na test skupu: {test_acc:.4f}")
  
    print("\nGeneriranje matrica zabune...")
    y_train_pred = np.argmax(best_model.predict(x_train), axis=1)
    y_test_pred = np.argmax(best_model.predict(x_test), axis=1)
    
    plot_confusion_matrix(y_train, y_train_pred, "Matrica zabune - Trening skup")
    plot_confusion_matrix(y_test, y_test_pred, "Matrica zabune - Test skup")

if __name__ == "__main__":
    main()
