#zadatak 2.
incorrect_test = np.where(np.argmax(y_pred_test, axis=1) != np.argmax(y_test, axis=1))[0]

for i in range(5):
    idx = incorrect_test[i]
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    true_label = np.argmax(y_test[idx])
    pred_label = np.argmax(y_pred_test[idx])
    plt.title(f'True: {true_label} Pred: {pred_label}')
    plt.axis('off')
    plt.show()
