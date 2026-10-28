import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X = np.load("X.npy")
y = np.load("y.npy")

xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=9)

clf = MLPClassifier(hidden_layer_sizes=(300,), alpha=0.01,
                    batch_size=256, max_iter=500)
clf.fit(xtr, ytr)

print("Accuracy:", round(accuracy_score(yte, clf.predict(xte)) * 100, 1), "%")