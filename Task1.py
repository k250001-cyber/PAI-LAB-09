import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

hours = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0])
marks = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])

model = LinearRegression()
model.fit(hours.reshape(-1, 1), marks)

print("Slope     :", round(model.coef_[0], 4))
print("Intercept :", round(model.intercept_, 4))

for h in [5.5, 8]:
    print("Predicted marks for", h, "hours:", round(model.predict([[h]])[0], 2))

y_hat = model.predict(hours.reshape(-1, 1))

plt.figure(figsize=(7, 5))
plt.scatter(hours, marks, label="Actual data")
plt.plot(hours, y_hat, color="red", label="Regression line")
plt.xlabel("Hours studied")
plt.ylabel("Marks obtained")
plt.title("Study Hours vs Marks")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("MAE :", round(mean_absolute_error(marks, y_hat), 4))
print("MSE :", round(mean_squared_error(marks, y_hat), 4))
print("R2  :", round(r2_score(marks, y_hat), 4))
