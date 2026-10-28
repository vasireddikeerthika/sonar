import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Xr, yr, ar = np.load("Xw.npy"), np.load("yw.npy"), np.load("actorsw.npy")
Xc, yc, ac = np.load("Xwc.npy"), np.load("ywc.npy"), np.load("actorswc.npy")

keep = ~np.isin(yr, ["calm", "surprised"])
Xr, yr, ar = Xr[keep], yr[keep], ar[keep]

def fit_eval(Xtr, ytr, Xte, yte, label):
    m = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, ytr)
    print(label, round(accuracy_score(yte, m.predict(Xte)) * 100, 1), "%", flush=True)

rte, cte = ar > 18, ac >= 1076

scr = StandardScaler().fit(Xr[~rte])
scc = StandardScaler().fit(Xc[~cte])
Xr_tr, Xr_te = scr.transform(Xr[~rte]), scr.transform(Xr[rte])
Xc_tr, Xc_te = scc.transform(Xc[~cte]), scc.transform(Xc[cte])

fit_eval(Xr_tr, yr[~rte], Xr_te, yr[rte], "A: RAVDESS only -> RAVDESS test   ")
fit_eval(Xc_tr, yc[~cte], Xc_te, yc[cte], "B: CREMA-D only -> CREMA-D test   ")

Xb = np.vstack([Xr_tr, Xc_tr])
yb = np.concatenate([yr[~rte], yc[~cte]])
fit_eval(Xb, yb, Xr_te, yr[rte], "1: Combined -> RAVDESS test     ")
fit_eval(Xb, yb, Xc_te, yc[cte], "2: Combined -> CREMA-D test     ")

Xr_all = StandardScaler().fit_transform(Xr)
Xc_all = StandardScaler().fit_transform(Xc)
fit_eval(Xc_all, yc, Xr_all, yr, "3: Train CREMA-D -> test RAVDESS ")
fit_eval(Xr_all, yr, Xc_all, yc, "4: Train RAVDESS -> test CREMA-D ")