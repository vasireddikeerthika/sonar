import numpy as np, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

Xr, yr = np.load("Xw.npy"), np.load("yw.npy")
Xc, yc = np.load("Xwc.npy"), np.load("ywc.npy")

keep = ~np.isin(yr, ["calm", "surprised"])
Xr, yr = Xr[keep], yr[keep]

X = np.vstack([Xr, Xc])
y = np.concatenate([yr, yc])

sc = StandardScaler().fit(X)
model = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X), y)

joblib.dump((sc, model), "sonar_w2v.joblib")
print("Saved. Classes:", model.classes_)