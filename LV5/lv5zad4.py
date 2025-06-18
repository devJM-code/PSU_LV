import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

DATA_PATH = 'occupancy_processed.csv'
FEATURES = ['S3_Temp', 'S5_CO2']
TARGET = 'Room_Occupancy_Count'
CLASS_NAMES = ['Slobodna', 'Zauzeta']

def load_and_prepare_data():
   
    df = pd.read_csv(DATA_PATH)
    
    print("\nOsnovne informacije o skupu podataka:")
    print(f"Broj primjera: {len(df)}")
    print(f"Distribucija klasa:\n{df[TARGET].value_counts()}")
    
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    
    return X, y

def visualize_data(X, y):

    plt.figure(figsize=(10, 6))
    
    for class_value, class_name in zip(np.unique(y), CLASS_NAMES):
        mask = y == class_value
        plt.scatter(X[mask, 0], X[mask, 1], 
                   label=class_name,
                   alpha=0.6)
    
    plt.xlabel('Temperatura (S3_Temp)')
    plt.ylabel('CO2 koncentracija (S5_CO2)')
    plt.title('Rasprostranjenost podataka po klasama')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_logistic_regression(X, y):
   
  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled, y)
   
    y_pred = model.predict(X_scaled)
   
    print("\nMatrica zabune:")
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap='Blues')
    plt.show()
    
    print("\nDetaljni izvještaj klasifikacije:")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES))
    
    return model, X_scaled

def analyze_decision_boundary(model, X, y):
   
    plt.figure(figsize=(10, 6))
   
    for class_value, class_name in zip(np.unique(y), CLASS_NAMES):
        mask = y == class_value
        plt.scatter(X[mask, 0], X[mask, 1], 
                   label=class_name,
                   alpha=0.6)
  
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.xlabel('Standardizirana temperatura')
    plt.ylabel('Standardizirana CO2 koncentracija')
    plt.title('Granica odluke logističke regresije')
    plt.legend()
    plt.show()

def main():
    """Glavni tok izvođenja"""
  
    X, y = load_and_prepare_data()
  
    visualize_data(X, y)
    
    model, X_scaled = train_logistic_regression(X, y)
   
    analyze_decision_boundary(model, X_scaled, y)
    
    print("\nZapažanja o modelu:")
    print("- Logistička regresija daje linearnu granicu odluke koja možda")
    print("  nije optimalna za ove podatke jer distribucija klasa nije")
    print("  linearno separabilna. Potrebno je razmotriti:")
    print("  * Dodatno inženjerstvo značajki")
    print("  * Nelinearne modele")
    print("  * Druge tehnike klasifikacije")

if __name__ == "__main__":
    main()
