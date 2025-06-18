import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True

def generate_true_function(x):
    """Generira pravu (nelinearnu) funkciju"""
    return (1.6345 - 0.6235 * np.cos(0.6067 * x) 
            - 1.3501 * np.sin(0.6067 * x) 
            - 1.1622 * np.cos(2 * x * 0.6067) 
            - 0.9443 * np.sin(2 * x * 0.6067))

def add_controlled_noise(y, noise_level=0.1, random_seed=14):
    """Dodaje šum podacima sa kontroliranim intenzitetom"""
    np.random.seed(random_seed)
    amplitude = np.max(y) - np.min(y)
    return y + noise_level * amplitude * np.random.normal(0, 1, len(y))

x_values = np.linspace(1, 10, 50)
y_true = generate_true_function(x_values)
y_noisy = add_controlled_noise(y_true)

x_values = x_values[:, np.newaxis]
y_noisy = y_noisy[:, np.newaxis]

def prepare_polynomial_features(x, degree=15):
    """Kreira polinomske feature-e do određenog stupnja"""
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(x)

x_poly = prepare_polynomial_features(x_values)

def train_test_split(x, y, test_size=0.3, random_seed=12):
    """Podjela podataka na trening i test skup"""
    np.random.seed(random_seed)
    n_samples = len(x)
    indices = np.random.permutation(n_samples)
    split_idx = int(np.floor((1 - test_size) * n_samples))
    
    return (x[indices[:split_idx]], y[indices[:split_idx]], 
            x[indices[split_idx:]], y[indices[split_idx:]])

x_train, y_train, x_test, y_test = train_test_split(x_poly, y_noisy)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")

def plot_results():
    """Prikazuje rezultate modela"""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.scatter(x_test[:, 1], y_test, c='red', label='Stvarna vrijednost')
    ax1.scatter(x_test[:, 1], y_pred, c='green', label='Predviđena vrijednost')
    ax1.set_title('Usporedba stvarnih i predviđenih vrijednosti')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
  
    ax2.plot(x_values, y_true, 'b-', label='Prava funkcija')
    ax2.plot(x_values, model.predict(x_poly), 'r-', label='Model')
    ax2.scatter(x_train[:, 1], y_train, c='black', label='Trening podaci')
    ax2.set_title('Ponašanje modela na cijelom rasponu')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

plot_results()
