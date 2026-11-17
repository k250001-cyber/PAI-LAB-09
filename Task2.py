import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0])
y = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])


def gradient_descent(x, y, lr, epochs):
    m, c = 0.0, 0.0
    n = len(x)
    history = []
    for epoch in range(1, epochs + 1):
        error = (m * x + c) - y
        dm = (2 / n) * np.sum(error * x)
        dc = (2 / n) * np.sum(error)
        m = m - lr * dm
        c = c - lr * dc
        mse = np.mean((y - (m * x + c)) ** 2)
        history.append(mse)
        if epoch % 100 == 0:
            print("  Epoch", epoch, "-> MSE =", round(mse, 4))
    return m, c, history


epochs = 2000
results = {}
for lr in [0.001, 0.01, 0.05]:
    print("Learning rate:", lr)
    m, c, history = gradient_descent(x, y, lr, epochs)
    results[lr] = (m, c, history)
    print("  Final m =", round(m, 4), ", c =", round(c, 4))

plt.figure(figsize=(8, 5))
for lr, (_, _, history) in results.items():
    plt.plot(range(1, epochs + 1), history, label="lr = " + str(lr))
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Loss Curve for Different Learning Rates")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

sk = LinearRegression().fit(x.reshape(-1, 1), y)
print()
print("scikit-learn  : m =", round(sk.coef_[0], 4), ", c =", round(sk.intercept_, 4))
for lr, (m, c, _) in results.items():
    print("GD lr =", lr, ": m =", round(m, 4), ", c =", round(c, 4))
