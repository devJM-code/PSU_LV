import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error

def non_func(x):
    y = (1.6345
         - 0.6235 * np.cos(0.6067 * x)
         - 1.3501 * np.sin(0.6067 * x)
         - 1.1622 * np.cos(2 * x * 0.6067)
         - 0.9443 * np.sin(2 * x * 0.6067))
    return y

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1 * varNoise * np.random.normal(0, 1, len(y))
    return y_noisy

x = np.linspace(1, 10, 100)
y_true = non_func(x)
y_measured = add_noise(y_true)

plt.figure(1)
plt.plot(x, y_measured, 'ok', label='Mjereno')
plt.plot(x, y_true, label='Stvarno')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.title('Stvarna funkcija i mjerenja s bukom')
plt.show()

np.random.seed(12)
indeksi = np.random.permutation(len(x))
granica = int(np.floor(0.7 * len(x)))
indeksi_train = indeksi[:granica]
indeksi_test = indeksi[granica+1:]

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]
xtrain = x[indeksi_train]
ytrain = y_measured[indeksi_train]
xtest = x[indeksi_test]
ytest = y_measured[indeksi_test]

plt.figure(2)
plt.plot(xtrain, ytrain, 'ob', label='Treniranje')
plt.plot(xtest, ytest, 'or', label='Testiranje')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.title('Podjela podataka: treniranje vs testiranje')
plt.show()

linearModel = lm.LinearRegression()
linearModel.fit(xtrain, ytrain)

print('Model je oblika: y_hat = theta0 + theta1 * x')
print('theta0 (intercept):', linearModel.intercept_)
print('theta1 (koeficijent):', linearModel.coef_)

ytest_p = linearModel.predict(xtest)
MSE_test = mean_squared_error(ytest, ytest_p)
print('Srednja kvadratna pogreška (MSE) na test skupu:', round(MSE_test, 4))

plt.figure(3)
plt.plot(xtest, ytest, 'or', label='Test podaci')
plt.plot(xtest, ytest_p, 'og', label='Predikcija modela')

x_line = np.array([1, 10])[:, np.newaxis]
y_line = linearModel.predict(x_line)
plt.plot(x_line, y_line, 'b-', label='Regresijska linija')

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.title('Predikcija test skupa i regresijska linija')
plt.show()
