import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, 
                           r2_score, 
                           mean_squared_error, 
                           max_error)

def load_and_prepare_data(file_path):

    data = pd.read_csv(file_path)
    print("\nInformacije o skupu podataka:")
    print(data.info())
    
    data = data.drop("name", axis=1)
    return data

def prepare_features_and_target(data):

    features = data[['km_driven', 'year', 'engine', 'max_power']]
    target = data['selling_price']
    return features, target

def train_and_evaluate_model(X, y, test_size=0.2, random_state=300):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    

    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)
    
    return model, y_test, test_predictions, y_train, train_predictions

def evaluate_performance(y_true, y_pred, prefix=""):

    print(f"\n{prefix}Performanse modela:")
    print(f"R2 score: {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"Max error: {max_error(y_true, y_pred):.2f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")

def plot_results(y_true, y_pred):

    plt.figure(figsize=(10, 8))
    ax = sns.regplot(x=y_pred, y=y_true, 
                    line_kws={'color': 'green', 'alpha': 0.5},
                    scatter_kws={'alpha': 0.6})
    
    ax.set(xlabel='Predviđene vrijednosti', 
          ylabel='Stvarne vrijednosti',
          title='Usporedba stvarnih i predviđenih cijena automobila')
    plt.grid(True, alpha=0.3)
    plt.show()

def main():

    df = load_and_prepare_data('cars_processed.csv')
    
    X, y = prepare_features_and_target(df)

    model, y_test, y_pred_test, _, _ = train_and_evaluate_model(X, y)
    
    evaluate_performance(y_test, y_pred_test, "Testni skup - ")
    
    y_pred_rupee = np.exp(y_pred_test)
    y_test_rupee = np.exp(y_test)
    
    print("\nPerformanse nakon eksponencijalne transformacije:")
    print(f"TRUE RMSE: {np.sqrt(mean_squared_error(y_test_rupee, y_pred_rupee)):.2f}")
    print(f"TRUE MAE: {mean_absolute_error(y_test_rupee, y_pred_rupee):.2f}")
    
    plot_results(y_test, y_pred_test)

if __name__ == "__main__":
    main()
