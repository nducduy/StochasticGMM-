"""Wall-clock and memory comparison: compiled single-pass SGMM versus offline two-step GMM (NumPy/BLAS).

usage: python3 timing.py sim      (simulated exponential IV model, n = 1e5, 1e6, 1e7)
       python3 timing.py check    (C implementation agrees with the Python reference)
"""
import ctypes, os, sys, time, json, subprocess
import numpy as np
from sgmm import ExpIV, chunk, gn_gmm, offline_two_step

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "libsgmm.so")
if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(os.path.join(HERE, "sgmm_c.c")):
    subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", LIB, os.path.join(HERE, "sgmm_c.c"), "-lm"])
lib = ctypes.CDLL(LIB)
P = ctypes.POINTER(ctypes.c_double)
lib.sgmm_exp.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_int, P, P, P, ctypes.c_long, ctypes.c_double,
                         ctypes.c_double, ctypes.c_double, ctypes.c_double] + [P] * 11
ptr = lambda a: a.ctypes.data_as(P)


def model_r(x, y, b):
    e = y * np.exp(-(x @ b)); return e - 1.0, -e[:, None] * x


def pilot_init(z, x, y, n0, eta0=1e-6):
    """2SLS-weighted GMM on the first n0 observations and the initial Phi, W, H (Algorithm 1)."""
    class M: r = staticmethod(lambda x, y, b: model_r(x, y, b))
    z0, x0, y0 = z[:n0], x[:n0], y[:n0]
    W1 = np.linalg.inv(z0.T @ z0 / n0)
    b, g, G = gn_gmm(M, z0, x0, y0, np.zeros(x.shape[1]), W1)
    r, _ = model_r(x0, y0, b); gz = z0 * r[:, None]
    S0 = gz.T @ gz / n0 + eta0 * np.eye(z.shape[1]); W0 = np.linalg.inv(S0)
    H0 = np.linalg.inv(G.T @ W0 @ G)
    return b, G, W0, H0


def sgmm_c(z, x, y, n0, gamma0=1.0, a=0.75, i0=100.0, box=10.0, return_rs=False):
    """Runs the compiled SGMM on observations n0, n0+1, ... after the pilot; returns bbar, H/n and J."""
    b0, Phi, W, H = pilot_init(z, x, y, n0)
    zs, xs, ys = (np.ascontiguousarray(v[n0:]) for v in (z, x, y)); n = len(ys); dg, db = z.shape[1], x.shape[1]
    beta = b0.copy(); bbar = b0.copy(); Phi = np.ascontiguousarray(Phi); W = np.ascontiguousarray(W); H = np.ascontiguousarray(H)
    Sg = np.zeros(dg); SGb = np.zeros(dg); SG = np.zeros((dg, db)); A0 = np.zeros(1); A1 = np.zeros(db); A2 = np.zeros((db, db))
    lib.sgmm_exp(n, dg, db, ptr(zs), ptr(xs), ptr(ys), n0, gamma0, a, i0, box,
                 ptr(beta), ptr(bbar), ptr(Phi), ptr(W), ptr(H), ptr(Sg), ptr(SGb), ptr(SG), ptr(A0), ptr(A1), ptr(A2))
    ghat = (Sg - SGb + SG @ bbar) / n
    VRS = (A2 - np.outer(A1, bbar) - np.outer(bbar, A1) + A0[0] * np.outer(bbar, bbar)) / n ** 2
    WP = W @ Phi; Jp = n * ghat @ (W - WP @ np.linalg.solve(Phi.T @ WP, WP.T)) @ ghat
    if return_rs:
        return bbar, H / n, Jp, VRS / n
    return bbar, H / n, Jp


class ExpM:
    r = staticmethod(lambda x, y, b: model_r(x, y, b))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "check":
        M = ExpIV(); z, x, y = M.gen(np.random.default_rng(1), 101_000)
        b, V, J = sgmm_c(z, x, y, 1000)
        print("C SGMM", b, np.sqrt(np.diag(V)), J)
        off = offline_two_step(ExpM, z, x, y, np.zeros(3)); print("offline", off[2], np.sqrt(np.diag(off[3])), off[4])
    if stage == "stream":
        # data arrive in batches of 10^5; offline GMM is re-estimated (warm start) after every batch
        M = ExpIV(); rng = np.random.default_rng(8); n = 1_000_000; z, x, y = M.gen(rng, n); n0 = 1000; out = {}
        t = time.perf_counter(); b, V, J = sgmm_c(z, x, y, n0, gamma0=2.0 * n0 ** -0.25, i0=float(n0)); out["online_total"] = time.perf_counter() - t
        bw = np.zeros(3); t = time.perf_counter(); per = []
        for k in range(1, 11):
            m = k * 100_000; t1 = time.perf_counter()
            off = offline_two_step(ExpM, z[:m], x[:m], y[:m], bw); bw = off[2]; per.append(time.perf_counter() - t1)
        out["offline_total"] = time.perf_counter() - t; out["offline_per_batch"] = per
        print(json.dumps(out)); json.dump(out, open(os.path.join(HERE, "res_timing_stream.json"), "w"), indent=1)
    if stage == "sim":
        M = ExpIV(); out = {}
        for n in [100_000, 1_000_000, 10_000_000]:
            rng = np.random.default_rng(7); parts = [M.gen(rng, min(n, 1_000_000)) for _ in range(max(1, n // 1_000_000))]
            z = np.vstack([p[0] for p in parts]); x = np.vstack([p[1] for p in parts]); y = np.concatenate([p[2] for p in parts])
            n0 = 1000
            ts = []
            for rep in range(3):
                t = time.perf_counter(); b, V, J = sgmm_c(z, x, y, n0, gamma0=2.0 * n0 ** -0.25, i0=float(n0)); ts.append(time.perf_counter() - t)
            to = []
            for rep in range(3):
                t = time.perf_counter(); off = offline_two_step(ExpM, z, x, y, np.zeros(3)); to.append(time.perf_counter() - t)
            mem_off = (z.nbytes + x.nbytes + y.nbytes) / 2 ** 20
            mem_on = 8 * (6 * 6 + 6 * 3 + 3 * 3 + 4 * 3 + 2 * 6 + 6 * 3) / 2 ** 10
            out[n] = dict(t_online=min(ts), t_offline=min(to), mem_data_MB=mem_off, mem_online_KB=mem_on,
                          b_online=b.tolist(), b_offline=off[2].tolist(), per_obs_us=1e6 * min(ts) / n)
            print(n, json.dumps(out[n]), flush=True)
        json.dump(out, open(os.path.join(HERE, "res_timing.json"), "w"), indent=1)
