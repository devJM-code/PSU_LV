import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = 'occupancy_processed.csv'
FEATURES = ['S3_Temp', 'S5_CO2']
TARGET = 'Room_Occupancy_Count'
CLASS_NAMES = ['Slobodna', 'Zauzeta']

def load_and_analyze_data():
  
    df = pd.read_csv(DATA_PATH)
  
    print("\nOsnovne informacije o skupu podataka:")
    print(df.info())
    
    print(f"\nBroj podatkovnih primjera: {len(df)}")
    print("\nRazdioba po klasama:")
    print(df[TARGET].value_counts())
    
    return df

def visualize_data(X, y):
    plt.figure(figsize=(10, 6))
    
    for class_value in np.unique(y):
        mask = y == class_value
        plt.scatter(X[mask, 0], X[mask, 1], 
                   label=CLASS_NAMES[class_value],
                   alpha=0.6)
    
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('CO2 koncentracija (ppm)')
    plt.title('Odnos temperature i CO2 koncentracije prema zauzetosti prostorije')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():

    df = load_and_analyze_data()

    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    visualize_data(X, y)

if __name__ == "__main__":
    main()
