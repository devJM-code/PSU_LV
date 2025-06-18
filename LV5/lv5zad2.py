import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, 
                           classification_report, 
                           ConfusionMatrixDisplay,
                           accuracy_score,
                           precision_score,
                           recall_score)

DATA_PATH = 'occupancy_processed.csv'
FEATURES = ['Temperature', 'CO2']
TARGET = 'Occupancy'
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_NEIGHBORS = 5

def load_and_prepare_data():

    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES].values
    y = data[TARGET].values
    return X, y

def split_and_scale_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        stratify=y, 
        random_state=RANDOM_STATE
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_knn_model(X_train, y_train, n_neighbors=N_NEIGHBORS):
  
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def evaluate_model(model, X_test, y_test):
   
    y_pred = model.predict(X_test)
    
  
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matrica zabune')
    plt.show()
    

    print("\nDetaljna evaluacija modela:")
    print(classification_report(y_test, y_pred))
    
  
    print(f"Točnost: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Preciznost: {precision_score(y_test, y_pred):.4f}")
    print(f"Odziv: {recall_score(y_test, y_pred):.4f}")

def analyze_parameter_impact(X_train, X_test, y_train, y_test):
    print("\nAnaliza utjecaja parametara:")
    
 
    neighbors = [1, 3, 5, 10, 20]
    for n in neighbors:
        model = train_knn_model(X_train, y_train, n)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Točnost za k={n}: {acc:.4f}")
    

    X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    model = train_knn_model(X_train_unscaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_unscaled))
    print(f"\nTočnost bez skaliranja: {acc:.4f}")

def main():
    
    X, y = load_and_prepare_data()
    
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)
    
    knn_model = train_knn_model(X_train, y_train)
    
    evaluate_model(knn_model, X_test, y_test)
  
    analyze_parameter_impact(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
