"""Empirical application: effect of a third child on weeks worked (Angrist and Evans, 1998),
multiplicative exponential IV model estimated by offline two-step GMM and by single-pass SGMM.

Data: AER package data set 'Fertility' (1980 US Census, women aged 21-35 with at least two children),
CSV from the Rdatasets collection (csv/AER/Fertility.csv), n = 254,654.
"""
import json, sys, time
import numpy as np, pandas as pd
from sgmm import gn_gmm
from timing import sgmm_c, model_r, ExpM

PATH = "/home/claude/data/rd/csv/AER/Fertility.csv"
d = pd.read_csv(PATH)
yes = lambda c: (d[c] == "yes").astype(float).values
male1 = (d.gender1 == "male").astype(float).values; male2 = (d.gender2 == "male").astype(float).values
agec = (d.age.values - 30.0) / 5.0
y = d.work.values.astype(float)
X = np.column_stack([np.ones(len(d)), yes("morekids"), male1, agec, agec ** 2, yes("afam"), yes("hispanic"), yes("other")])
Z = np.column_stack([np.ones(len(d)), male1 * male2, (1 - male1) * (1 - male2), male1, agec, agec ** 2,
                     yes("afam"), yes("hispanic"), yes("other")])
names = ["const", "morekids", "boy1st", "age", "age2", "afam", "hispanic", "other"]
n = len(y); out = {"n": n, "mean_work": float(y.mean()), "share_zero": float((y == 0).mean()),
                   "share_morekids": float(X[:, 1].mean())}

# ---- linear 2SLS with the two instruments (benchmark, Angrist-Evans style)
Pz = np.linalg.solve(Z.T @ Z, Z.T @ X); Xh = Z @ Pz
b2 = np.linalg.solve(Xh.T @ X, Xh.T @ y); u = y - X @ b2
A = np.linalg.inv(Xh.T @ X); V2 = A @ (Xh.T * u ** 2) @ Xh @ A.T
out["lin2sls_morekids"] = [float(b2[1]), float(np.sqrt(V2[1, 1]))]

# ---- offline two-step GMM (full data)
t = time.perf_counter()
W1 = np.linalg.inv(Z.T @ Z / n)
b1, _, _ = gn_gmm(ExpM, Z, X, y, np.zeros(8), W1)
r, _ = model_r(X, y, b1); gz = Z * r[:, None]; W2 = np.linalg.inv(gz.T @ gz / n)
bo, go, Go = gn_gmm(ExpM, Z, X, y, b1, W2)
Vo = np.linalg.inv(Go.T @ W2 @ Go) / n; Jo = n * go @ W2 @ go
t_off = time.perf_counter() - t
out["offline"] = dict(b=bo.tolist(), se=np.sqrt(np.diag(Vo)).tolist(), J=float(Jo), time=t_off)

# ---- SGMM, one pass in each of 20 random orders (pilot = first 5000 observations of the order)
n0 = 5000; runs = []
for s in range(20):
    perm = np.random.default_rng(100 + s).permutation(n)
    z, x, yy = (np.ascontiguousarray(v[perm]) for v in (Z, X, y))
    t = time.perf_counter(); b, V, J, VRS = sgmm_c(z, x, yy, n0, gamma0=2.0 * n0 ** (-0.25), a=0.75, i0=float(n0), return_rs=True); tt = time.perf_counter() - t
    runs.append(dict(b=b.tolist(), se=np.sqrt(np.diag(V)).tolist(), se_rs=np.sqrt(np.diag(VRS)).tolist(), J=float(J), time=tt))
out["sgmm"] = runs
B = np.array([r_["b"] for r_ in runs])
out["sgmm_order_sd"] = B.std(0).tolist()
json.dump(out, open("res_application.json", "w"), indent=1)

print("n", n, "mean weeks", y.mean(), "zero share", (y == 0).mean(), "morekids", X[:, 1].mean())
print("linear 2SLS morekids", b2[1], np.sqrt(V2[1, 1]))
for k, nm in enumerate(names):
    print(f"{nm:9s} off {bo[k]: .4f} ({np.sqrt(Vo[k, k]):.4f}) | sgmm[0] {runs[0]['b'][k]: .4f} ({runs[0]['se'][k]:.4f}) "
          f"| mean over orders {B[:, k].mean(): .4f}, sd {B[:, k].std():.4f}")
print("J offline", Jo, "J online", [round(r_["J"], 2) for r_ in runs[:5]], "time off", t_off, "time sgmm", np.mean([r_["time"] for r_ in runs]))
