import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    "experience": [1, 2, 3, None, 5, 6, 7, 8, 9, 10, 3, 5],
    "education": ["BS", "BS", "MS", "BS", "MS", "PhD", "MS", "PhD", "BS", "PhD", "BS", "MS"],
    "salary_k": [45, 50, 70, 58, 85, 120, 95, 140, 90, 160, 55, 88],
})

median_exp = df["experience"].median()
df["experience"] = df["experience"].fillna(median_exp)
print("Median used to fill missing value:", median_exp)

df = pd.get_dummies(df, columns=["education"], dtype=int)
print("Columns after one-hot encoding:", list(df.columns))

X = df.drop("salary_k", axis=1)
y = df["salary_k"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
print("Test MAE :", round(mean_absolute_error(y_test, y_pred), 4))
print("Test R2  :", round(r2_score(y_test, y_pred), 4))

new_grad = pd.DataFrame([{"experience": 4, "education_BS": 0, "education_MS": 1, "education_PhD": 0}])[X.columns]
print("Predicted salary (k) for MS graduate with 4 years experience:",
      round(model.predict(scaler.transform(new_grad))[0], 2))
