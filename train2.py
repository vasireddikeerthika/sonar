import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X = np.load("X.npy")
y = np.load("y.npy")
a = np.load("actors.npy")

tr, te = a <= 18, a > 18          # actors 1-18 train, 19-24 test

sc = StandardScaler().fit(X[tr])  # fit on train only
xtr, xte = sc.transform(X[tr]), sc.transform(X[te])

clf = MLPClassifier(hidden_layer_sizes=(300,), alpha=0.01,
                    batch_size=256, max_iter=500)
clf.fit(xtr, y[tr])

print("Honest accuracy:", round(accuracy_score(y[te], clf.predict(xte)) * 100, 1), "%")
from sklearn.metrics import confusion_matrix, classification_report
pred = clf.predict(xte)
print(classification_report(y[te], pred))
print(confusion_matrix(y[te], pred))