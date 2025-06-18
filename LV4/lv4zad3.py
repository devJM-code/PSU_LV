import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(14)  
DEGREE = 15
TEST_SIZE = 0.3

def generate_data():
    x = np.linspace(1, 10, 50)
    
    y_true = (1.6345 - 0.6235 * np.cos(0.6067 * x) 
              - 1.3501 * np.sin(0.6067 * x) 
              - 1.1622 * np.cos(2 * x * 0.6067) 
              - 0.9443 * np.sin(2 * x * 0.6067))
    
    noise_level = 0.1 * (np.max(y_true) - np.min(y_true))
    y_noisy = y_true + noise_level * np.random.normal(0, 1, len(x))
    
    return x[:, None], y_true, y_noisy[:, None]

def prepare_model(x, y):
    poly = PolynomialFeatures(degree=DEGREE)
    x_poly = poly.fit_transform(x)
    
    np.random.seed(12)
    n_samples = len(x_poly)
    indices = np.random.permutation(n_samples)
    split_idx = int((1 - TEST_SIZE) * n_samples)
    
    x_train, y_train = x_poly[indices[:split_idx]], y[indices[:split_idx]]
    x_test, y_test = x_poly[indices[split_idx:]], y[indices[split_idx:]]
    
    model = LinearRegression().fit(x_train, y_train)
    return model, x_poly, x_test, y_test

def plot_results(x, y_true, model, x_poly, x_test, y_test):
    
    y_pred = model.predict(x_test)
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(x_test[:, 1], y_test, c='red', label='Stvarne vrijednosti')
    plt.scatter(x_test[:, 1], y_pred, c='green', label='Predikcije')
    plt.title('Usporedba stvarnih i predviđenih vrijednosti')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, y_true, 'b-', label='Istinita funkcija')
    plt.plot(x, model.predict(x_poly), 'r-', label='Model')
    plt.scatter(x_poly[indices[:split_idx], 1], y[indices[:split_idx]], 
                c='black', label='Trening podaci')
    plt.title('Ponašanje modela na cijelom rasponu')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
  
    mse = mean_squared_error(y_test, y_pred)
    print(f'Test MSE: {mse:.6f}')

x, y_true, y_noisy = generate_data()
model, x_poly, x_test, y_test = prepare_model(x, y_noisy)
plot_results(x, y_true, model, x_poly, x_test, y_test)
