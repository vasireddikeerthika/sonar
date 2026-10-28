import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X = np.load("Xw.npy")
y = np.load("yw.npy")
a = np.load("actorsw.npy")

accs = []
for tr, te in GroupKFold(n_splits=6).split(X, y, groups=a):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    m.fit(X[tr], y[tr])
    accs.append(m.score(X[te], y[te]))

print([round(x * 100, 1) for x in accs])
print("Mean:", round(np.mean(accs) * 100, 1), "+/-", round(np.std(accs) * 100, 1))