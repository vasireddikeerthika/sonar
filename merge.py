import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

Xr, yr, ar = np.load("X.npy"), np.load("y.npy"), np.load("actors.npy")
Xc, yc, ac = np.load("X_c.npy"), np.load("y_c.npy"), np.load("actors_c.npy")

keep = ~np.isin(yr, ["calm", "surprised"])
Xr, yr, ar = Xr[keep], yr[keep], ar[keep]

def run(Xtr, ytr, Xte, yte, label):
    sc = StandardScaler().fit(Xtr)
    m = SVC(kernel="rbf", C=10).fit(sc.transform(Xtr), ytr)
    acc = accuracy_score(yte, m.predict(sc.transform(Xte)))
    print(label, round(acc * 100, 1), "%")

# held-out test: RAVDESS actors 19-24 (same as before, 6 emotions)
te = ar > 18
Xte, yte = Xr[te], yr[te]

# A: train on RAVDESS only
run(Xr[~te], yr[~te], Xte, yte, "A: RAVDESS only      ->")

# B: train on RAVDESS + CREMA-D
Xb = np.vstack([Xr[~te], Xc])
yb = np.concatenate([yr[~te], yc])
run(Xb, yb, Xte, yte, "B: RAVDESS + CREMA-D ->")

# C: CREMA-D only, tested on CREMA-D actors 1076-1091 (held out)
tc = ac >= 1076
run(Xc[~tc], yc[~tc], Xc[tc], yc[tc], "C: CREMA-D only, own test ->")