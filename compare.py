import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = np.load("X.npy")
y = np.load("y.npy")
a = np.load("actors.npy")

tr, te = a <= 18, a > 18
sc = StandardScaler().fit(X[tr])
xtr, xte = sc.transform(X[tr]), sc.transform(X[te])

models = {
    "MLP": MLPClassifier(hidden_layer_sizes=(300,), alpha=0.01, batch_size=256, max_iter=500),
    "SVM": SVC(kernel="rbf", C=10),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=0),
}

for name, m in models.items():
    m.fit(xtr, y[tr])
    acc = accuracy_score(y[te], m.predict(xte))
    print(name, round(acc * 100, 1), "%")