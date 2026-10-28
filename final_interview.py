import numpy as np, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MAP = {"fearful": "nervous", "neutral": "flat", "sad": "flat",
       "happy": "confident", "angry": "intense"}

def load(X, y):
    keep = np.isin(y, list(MAP))
    return X[keep], np.array([MAP[v] for v in y[keep]])

Xr, yr = load(np.load("Xw.npy"), np.load("yw.npy"))
Xc, yc = load(np.load("Xwc.npy"), np.load("ywc.npy"))

X = np.vstack([Xr, Xc])
y = np.concatenate([yr, yc])

sc = StandardScaler().fit(X)
model = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(sc.transform(X), y)

joblib.dump((sc, model), "sonar_w2v.joblib")
print("Saved. Classes:", model.classes_)