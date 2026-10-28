import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

y = np.load("yw.npy")
a = np.load("actorsw.npy")
tr, te = a <= 18, a > 18

def run(X, tag):
    sc = StandardScaler().fit(X[tr])
    xtr, xte = sc.transform(X[tr]), sc.transform(X[te])
    models = {
        "SVM": SVC(kernel="rbf", C=10),
        "LogReg": LogisticRegression(max_iter=2000, C=0.1),
        "MLP": MLPClassifier(hidden_layer_sizes=(300,), alpha=0.01,
                             batch_size=256, max_iter=500),
    }
    for n, m in models.items():
        m.fit(xtr, y[tr])
        print(tag, n, round(accuracy_score(y[te], m.predict(xte)) * 100, 1), "%")

run(np.load("Xw.npy"), "wav2vec2 ")
run(np.load("X.npy"),  "handfeat ")