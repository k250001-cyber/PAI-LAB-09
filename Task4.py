import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X, y = data.data, data.target

print("Number of samples  :", X.shape[0])
print("Number of features :", X.shape[1])
counts = np.bincount(y)
for i, name in enumerate(data.target_names):
    print("Class", i, "(" + name + "):", counts[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Test accuracy:", round(model.score(X_test, y_test), 4))

proba = model.predict_proba(X_test)[:, 1]
for t in [0.3, 0.5, 0.7]:
    preds = (proba >= t).astype(int)
    print("Threshold", t, "-> predicted class 1:", int(preds.sum()),
          "| accuracy:", round(accuracy_score(y_test, preds), 4))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


manual = sigmoid(model.decision_function(X_test))
print("Matches predict_proba:", np.allclose(manual, proba))
print("Max absolute difference:", np.max(np.abs(manual - proba)))
