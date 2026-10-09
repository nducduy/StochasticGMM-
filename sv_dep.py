"""SGMM for dependent data (Section 5): delayed evaluation points and a long-run variance (Bartlett) weight,
applied to the stochastic volatility model of Example 2.3; offline benchmark: two-step GMM with the
same Bartlett HAC weight.

usage: python3 sv_dep.py R n_max D L [c]
  D = delay (0: no delay), L = Bartlett lag truncation (0: weight from E[g g'] as in Algorithm 1)
"""
import sys, json, time
import numpy as np
from sv import Stream, mm_rep, to_rep, from_rep, offline_rep, PLO, PHI, TH

R = int(sys.argv[1]); n_max = int(sys.argv[2]); D = int(sys.argv[3]); Lb = int(sys.argv[4])
c = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
n0 = 5000; a = 0.75; gamma0 = c * n0 ** (a - 1); i0 = n0; Lc = 10_000
CK = [k for k in [10_000, 100_000, 1_000_000] if k <= n_max]
wts = 1.0 - np.arange(1, Lb + 1) / (Lb + 1.0)          # Bartlett weights w_1..w_L


def hac(X):
    """Bartlett HAC covariance of the rows of X (n, d) with lag truncation Lb."""
    Xc = X - X.mean(0); S = Xc.T @ Xc / len(X)
    for h in range(1, Lb + 1):
        Gh = Xc[h:].T @ Xc[:-h] / len(X); S += wts[h - 1] * (Gh + Gh.T)
    return S


st = Stream(R, 2024)
X0 = st.next(n0)                                        # (n0, R, 24) pilot
p0 = to_rep(np.array([-1.0, 0.8, 0.3]))
S0 = np.array([hac(X0[:, r]) for r in range(R)])
S0iid = np.array([np.cov(X0[:, r].T, bias=True) for r in range(R)])
b0 = np.array([offline_rep(X0[:, r].mean(0), S0iid[r], p0) for r in range(R)])   # pilot estimate with i.i.d. weights (more stable)
M0, D0 = mm_rep(b0); Phi = -D0; W = np.linalg.inv(S0)
beta = b0.copy(); bbar = b0.copy()
hist_b = [b0.copy() for _ in range(D + 1)]             # beta_{i-1-D..i-1}
hist_bb = [b0.copy() for _ in range(D + 1)]
hist_P = [np.einsum("rij,rkj,rkl->ril", np.linalg.inv(np.einsum("rki,rkl,rlj->rij", Phi, W, Phi)), Phi, W)] * (D + 1)
s_hist = np.zeros((Lb + 1, R, 24))                      # s_{i-1}, ..., s_{i-Lb-1}
cwin = np.zeros((R, 24))                                # s_{i-1} + ... + s_{i-Lb-1} (window sum)
# running sums for the offline HAC estimator over pilot + stream
sx = X0.sum(0); tail = X0[-Lb:] if Lb else None
acc = np.einsum("nri,nrj->rij", X0, X0); acch = [np.einsum("nri,nrj->rij", X0[h:], X0[:-h]) for h in range(1, Lb + 1)]
res = {}; t0 = time.perf_counter(); nproj = 0; nskip = 0
for ci in range(n_max // Lc):
    Xc = st.next(Lc)
    # offline sums (exact lagged products across chunk boundaries)
    sx += Xc.sum(0); acc += np.einsum("nri,nrj->rij", Xc, Xc)
    if Lb:
        Xext = np.concatenate([tail, Xc], 0)
        for r in range(R):
            Xr = np.ascontiguousarray(Xext[:, r])
            for h in range(1, Lb + 1):
                acch[h - 1][r] += Xr[Lb:].T @ Xr[Lb - h:len(Xr) - h]
        tail = Xext[-Lb:]
    for t in range(Lc):
        i = ci * Lc + t + 1; N = n0 + i - 1.0; x = Xc[t]
        gam = gamma0 * (i + i0) ** (-a)
        bd, Pd, bbd = hist_b[0], hist_P[0], hist_bb[0]        # delayed quantities (index i-1-D)
        g = x - mm_rep(bd)[0]
        bn = beta - gam * np.einsum("rij,rj->ri", Pd, g)
        if i <= 1200 and i % 50 == 0 and len(sys.argv) > 6: print(i, np.round(np.abs(beta - b0).max(0), 3), np.round(np.abs(np.einsum("rij,rj->ri", Pd, g)).max(0), 1), "condW %.1e" % np.linalg.cond(W).max(), "|Pd| %.1e" % np.abs(Pd).max(), "|g| %.1e" % np.abs(g).max(), flush=True)
        cl = np.clip(bn, PLO, PHI); nproj += int((cl != bn).any(1).sum()); beta = cl
        mb, Db = mm_rep(bbd); s = x - mb
        # long-run (Bartlett) weight in window form: S += c c'/(L+1), c = s_i + ... + s_{i-L}  (psd by construction)
        cwin = cwin + s - s_hist[-1] if Lb else s
        u = cwin / np.sqrt(Lb + 1.0)
        Wu = np.einsum("rij,rj->ri", W, u); l1 = N + (u * Wu).sum(1)
        W = (N + 1) / N * (W - Wu[:, :, None] * Wu[:, None, :] / l1[:, None, None])
        Phi = (N * Phi - Db) / (N + 1)
        H = np.linalg.inv(np.einsum("rki,rkl,rlj->rij", Phi, W, Phi))
        P = np.einsum("rij,rkj,rkl->ril", H, Phi, W)
        if Lb:
            s_hist = np.concatenate([s[None], s_hist[:-1]], 0)
        bbar = bbar + (beta - bbar) / i
        hist_b = hist_b[1:] + [beta.copy()]; hist_bb = hist_bb[1:] + [bbar.copy()]; hist_P = hist_P[1:] + [P]
    n = (ci + 1) * Lc
    if n in CK:
        m = n0 + n; xb = sx / m
        S = acc / m - np.einsum("ri,rj->rij", xb, xb)
        for h in range(1, Lb + 1):
            Gh = acch[h - 1] / m - np.einsum("ri,rj->rij", xb, xb); S = S + wts[h - 1] * (Gh + np.transpose(Gh, (0, 2, 1)))
        off = np.array([offline_rep(xb[r], S[r], b0[r]) for r in range(R)])
        e1 = from_rep(bbar) - TH; e2 = from_rep(off) - TH
        res[n] = dict(sgmm=from_rep(bbar).tolist(), off=from_rep(off).tolist(), H=H.tolist(), time=time.perf_counter() - t0)
        print(n, "SGMM rmse", np.round(np.sqrt((e1 ** 2).mean(0)), 4), "bias", np.round(e1.mean(0), 4),
              "| GMM rmse", np.round(np.sqrt((e2 ** 2).mean(0)), 4), "bias", np.round(e2.mean(0), 4),
              "proj", nproj, "skip", nskip, "%.0fs" % (time.perf_counter() - t0), flush=True)
res["pilot"] = from_rep(b0).tolist(); res["cfg"] = dict(D=D, L=Lb, c=c, n0=n0, a=a); res["nproj"] = nproj
json.dump(res, open(f"res_svdep_{n_max}_D{D}_L{Lb}_c{c}.json", "w"))
