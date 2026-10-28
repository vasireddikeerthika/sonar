import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

Xr, yr, ar = np.load("X.npy"), np.load("y.npy"), np.load("actors.npy")
Xc, yc, ac = np.load("X_c.npy"), np.load("y_c.npy"), np.load("actors_c.npy")

keep = ~np.isin(yr, ["calm", "surprised"])
Xr, yr, ar = Xr[keep], yr[keep], ar[keep]

def fit_eval(Xtr, ytr, Xte, yte, label):
    m = SVC(kernel="rbf", C=10).fit(Xtr, ytr)
    print(label, round(accuracy_score(yte, m.predict(Xte)) * 100, 1), "%")

# held-out actors
rte, cte = ar > 18, ac >= 1076

# each dataset scaled with its OWN scaler (fit on its train part only)
scr = StandardScaler().fit(Xr[~rte])
scc = StandardScaler().fit(Xc[~cte])
Xr_tr, Xr_te = scr.transform(Xr[~rte]), scr.transform(Xr[rte])
Xc_tr, Xc_te = scc.transform(Xc[~cte]), scc.transform(Xc[cte])

# 1: combined training, per-dataset scaling, test on each
Xb = np.vstack([Xr_tr, Xc_tr])
yb = np.concatenate([yr[~rte], yc[~cte]])
fit_eval(Xb, yb, Xr_te, yr[rte], "1: Combined -> RAVDESS test ")
fit_eval(Xb, yb, Xc_te, yc[cte], "2: Combined -> CREMA-D test ")

# cross-corpus (each dataset scaled by its own scaler, all data)
Xr_all = StandardScaler().fit_transform(Xr)
Xc_all = StandardScaler().fit_transform(Xc)
fit_eval(Xc_all, yc, Xr_all, yr, "3: Train CREMA-D -> test RAVDESS ")
fit_eval(Xr_all, yr, Xc_all, yc, "4: Train RAVDESS -> test CREMA-D ")