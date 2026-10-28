import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X = np.load("X.npy")
y = np.load("y.npy")
a = np.load("actors.npy")

accs = []
for tr, te in GroupKFold(n_splits=6).split(X, y, groups=a):
    m = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10))
    m.fit(X[tr], y[tr])
    accs.append(m.score(X[te], y[te]))

print([round(x * 100, 1) for x in accs])
print("Mean:", round(np.mean(accs) * 100, 1), "+/-", round(np.std(accs) * 100, 1))