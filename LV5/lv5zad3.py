import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (confusion_matrix, 
                           classification_report, 
                           ConfusionMatrixDisplay,
                           accuracy_score)

DATA_PATH = 'occupancy_processed.csv'
FEATURES = ['Temperature', 'CO2']
TARGET = 'Occupancy'
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_DEPTH = 2

def load_data():
   
    df = pd.read_csv(DATA_PATH)
    print(f"\nUčitano {len(df)} primjera")
    print("Distribucija klasa:")
    print(df[TARGET].value_counts())
    return df

def prepare_features(df, scale=True):
  
    X = df[FEATURES]
    y = df[TARGET]
    
    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)
        print("\nZnačajke su skalirane")
    else:
        print("\nZnačajke nisu skalirane")
    
    return X, y

def train_and_evaluate(X, y, max_depth=MAX_DEPTH):
  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        stratify=y, 
        random_state=RANDOM_STATE
    )
 
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
 
    y_pred = dt.predict(X_test)
    
    print("\nMatrica zabune:")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    
    print("\nIzvještaj klasifikacije:")
    print(classification_report(y_test, y_pred))
    
    return dt, X_train, y_train

def visualize_tree(model, feature_names, class_names):

    plt.figure(figsize=(12, 8))
    plot_tree(model, 
             feature_names=feature_names,
             class_names=class_names,
             filled=True,
             rounded=True)
    plt.title("Stablo odlučivanja")
    plt.show()

def analyze_depth_impact(X, y, depths=[1, 2, 3, 5, 10, None]):
  
    print("\nUtjecaj max_depth parametra:")
    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        dt.fit(X_train, y_train)
        acc = accuracy_score(y_test, dt.predict(X_test))
        print(f"max_depth={str(depth).ljust(4)} | Točnost: {acc:.4f}")

def main():
    """Glavni tok izvođenja"""
   
    df = load_data()
   
    X_scaled, y = prepare_features(df, scale=True)
   
    dt_model, X_train, y_train = train_and_evaluate(X_scaled, y)
   
    visualize_tree(dt_model, FEATURES, ['Slobodna', 'Zauzeta'])
   
    analyze_depth_impact(X_scaled, y)
 
    X_unscaled, y = prepare_features(df, scale=False)
    print("\nRezultati bez skaliranja značajki:")
    train_and_evaluate(X_unscaled, y)

if __name__ == "__main__":
    main()
