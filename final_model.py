import numpy as np, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

Xr, yr = np.load("X.npy"), np.load("y.npy")
Xc, yc = np.load("X_c.npy"), np.load("y_c.npy")

keep = ~np.isin(yr, ["calm", "surprised"])
Xr, yr = Xr[keep], yr[keep]

scr = StandardScaler().fit(Xr)
scc = StandardScaler().fit(Xc)

X = np.vstack([scr.transform(Xr), scc.transform(Xc)])
y = np.concatenate([yr, yc])

model = SVC(kernel="rbf", C=10, probability=True).fit(X, y)

# app scaler: fit on everything combined (a real mic is a new domain)
app_sc = StandardScaler().fit(np.vstack([Xr, Xc]))
X_app = app_sc.transform(np.vstack([Xr, Xc]))
model_app = SVC(kernel="rbf", C=10, probability=True).fit(X_app, y)

joblib.dump((app_sc, model_app), "sonar_model.joblib")
print("Saved. Classes:", model_app.classes_)