"""Stochastic (online) GMM for IV-type moment functions g(x,b) = z * r(x,b), with Jacobian G = z dr'.

All online algorithms are vectorised over R independent replications: every state array has a
leading axis of length R and observation i of every replication is processed at step i.
"""
import time
import numpy as np


# ----------------------------------------------------------------------------- models
class LinearIV:
    """y = x'b + u, x = (1, x1, x2) with x1, x2 endogenous, z = (1, z1..z7); u heteroskedastic with
    conditional scale exp((z1 + z4)/2 - 1/2)."""
    name = "linear"; d = 3; q = 8
    beta = np.array([1.0, 0.5, -0.5])

    def gen(self, rng, n):
        zz = rng.standard_normal((n, 7))
        v = rng.standard_normal((n, 2)); e = rng.standard_normal(n)
        x1 = 0.3 * zz[:, 0:3].sum(1) + v[:, 0]
        x2 = 0.3 * zz[:, 3:6].sum(1) + v[:, 1]
        rho = 0.5
        u = (rho * (v[:, 0] + v[:, 1]) / np.sqrt(2) + np.sqrt(1 - rho ** 2) * e) * np.exp(0.5 * (zz[:, 0] + zz[:, 3]) - 0.5)
        x = np.column_stack([np.ones(n), x1, x2]); z = np.column_stack([np.ones(n), zz])
        y = x @ self.beta + u
        return z, x, y

    @staticmethod
    def r(x, y, b):            # residual and its gradient in b;  g = z * r
        return (x * b).sum(-1) - y, x


class ExpIV:
    """Multiplicative exponential model y = exp(x'b) * eta, E[eta | z] = 1 (Mullahy, 1997);
    x = (1, x1, x2), x1 endogenous; z = (1, z1..z4, x2).  delta != 0 makes z4 an invalid instrument."""
    name = "exp"; d = 3; q = 6
    beta = np.array([0.5, 0.5, -0.5])

    def __init__(self, delta=0.0):
        self.delta = delta

    def gen(self, rng, n):
        zz = rng.standard_normal((n, 4)); x2 = rng.standard_normal(n)
        v = 0.5 * rng.standard_normal(n)
        w = 0.6 * v / 0.5 + 0.8 * rng.standard_normal(n)          # corr(v, w) = 0.6
        s = 0.1 + np.maximum(zz[:, 0] + zz[:, 1], 0.0)             # heteroskedasticity
        k = 1.0 / s ** 2
        eta = 0.5 * np.exp(0.5 * w - 0.125) + 0.5 * rng.gamma(k, 1.0 / k)   # E[eta | z] = 1
        eta = eta * np.exp(self.delta * zz[:, 3] - self.delta ** 2 / 2)    # invalid instrument if delta != 0
        x1 = 0.25 * zz.sum(1) + v
        x = np.column_stack([np.ones(n), x1, x2]); z = np.column_stack([np.ones(n), zz, x2])
        y = np.exp(x @ self.beta) * eta
        return z, x, y

    @staticmethod
    def r(x, y, b):
        e = y * np.exp(-(x * b).sum(-1))
        return e - 1.0, -e[..., None] * x


# ----------------------------------------------------------------------------- data streams
def chunk(model, seed, rep, c, L):
    """Chunk c (length L) of the data stream of replication rep; c = -1 is the pilot sample."""
    rng = np.random.default_rng([seed, rep, c + 1])
    return model.gen(rng, L)


def stream_chunk(model, seed, R, c, L):
    zs, xs, ys = zip(*(chunk(model, seed, r, c, L) for r in range(R)))
    return np.stack(zs, 1), np.stack(xs, 1), np.stack(ys, 1)     # (L, R, .)


# ----------------------------------------------------------------------------- offline GMM
def gn_gmm(model, z, x, y, b, W, iters=50, tol=1e-12):
    """Gauss-Newton for min gbar' W gbar on one data set (arrays (n,.)); damped by backtracking."""
    n = len(y)
    def obj(b):
        r, dr = model.r(x, y, b); g = z.T @ r / n; return g, z.T @ dr / n
    g, G = obj(b); f = g @ W @ g
    for _ in range(iters):
        step = np.linalg.solve(G.T @ W @ G, G.T @ W @ g)
        t = 1.0
        while True:
            bn = b - t * step; gn, Gn = obj(bn); fn = gn @ W @ gn
            if fn <= f or t < 1e-4: break
            t /= 2
        done = abs(f - fn) <= tol * max(f, 1e-300) or np.max(abs(bn - b)) < 1e-12
        b, g, G, f = bn, gn, Gn, fn
        if done: break
    return b, g, G


def offline_two_step(model, z, x, y, b0):
    n = len(y)
    W1 = np.linalg.inv(z.T @ z / n)
    b1, _, _ = gn_gmm(model, z, x, y, b0, W1)
    r, _ = model.r(x, y, b1); gz = z * r[:, None]
    W2 = np.linalg.inv(gz.T @ gz / n)
    b2, g2, G2 = gn_gmm(model, z, x, y, b1, W2)
    V2 = np.linalg.inv(G2.T @ W2 @ G2) / n
    # 2SLS-type (first step) sandwich variance
    G1 = z.T @ model.r(x, y, b1)[1] / n; A = np.linalg.inv(G1.T @ W1 @ G1) @ G1.T @ W1
    V1 = A @ (gz.T @ gz / n) @ A.T / n
    J = n * g2 @ W2 @ g2
    return b1, V1, b2, V2, J


def pilot(model, seed, R, n0, eta0=1e-6):
    """Pilot estimates on n0 observations per replication (2SLS-weighted GMM) and initial Phi, S."""
    out = []
    for r in range(R):
        z, x, y = chunk(model, seed, r, -1, n0)
        W1 = np.linalg.inv(z.T @ z / n0)
        b, g, G = gn_gmm(model, z, x, y, np.zeros(model.d), W1)
        res, _ = model.r(x, y, b); gz = z * res[:, None]
        out.append((b, G, gz.T @ gz / n0 + eta0 * np.eye(model.q), z.T @ z / n0 + eta0 * np.eye(model.q)))
    b, G, Sg, Sz = map(np.stack, zip(*out))
    return b, G, Sg, Sz


# ----------------------------------------------------------------------------- online SGMM
def run_online(model, seed, R, n_max, checkpoints, n0=1000, method="eff", gamma0=1.0, a=0.75,
               L=10_000, box=10.0, nu=0.1, cP=1e8, verbose=False, i0=100):
    """method: 'eff' (efficient SGMM, Gauss-Newton preconditioned), 's2sls' (online 2SLS weights),
    'first' (first-order: no (Phi'W Phi)^{-1} preconditioner, efficient weights)."""
    d, q = model.d, model.q
    b0, Phi, Sg, Sz = pilot(model, seed, R, n0)
    S = Sz if method == "s2sls" else Sg
    W = np.linalg.inv(S)
    H = np.linalg.inv(np.einsum("rki,rkl,rlj->rij", Phi, W, Phi))
    beta = b0.copy(); bbar = b0.copy()
    # accumulators for the online J statistic and the sandwich variance
    Sg_acc = np.zeros((R, q)); SGb = np.zeros((R, q)); SG = np.zeros((R, q, d)); Om = np.zeros((R, q, q))
    res = {}; n_proj = 0; n_trunc = 0; t0 = time.perf_counter()
    A2 = np.zeros((R, d, d)); A1 = np.zeros((R, d)); A0 = 0.0   # random-scaling sums
    I3 = np.eye(3)
    for c in range(n_max // L):
        Z, X, Y = stream_chunk(model, seed, R, c, L)
        for t in range(L):
            i = c * L + t + 1; N = n0 + i - 1.0
            z, x, y = Z[t], X[t], Y[t]
            gam = gamma0 * (i + i0) ** (-a)
            # ---- parameter step at the current iterate
            r_cur, _ = model.r(x, y, beta)
            g = z * r_cur[:, None]
            Wg = np.einsum("rij,rj->ri", W, g)
            u = np.einsum("rki,rk->ri", Phi, Wg)
            if method == "first":
                step = u
            else:
                step = np.einsum("rij,rj->ri", H, u)
                # truncation of the preconditioner (inactive in practice): ||P||_F <= ||H|| ||Phi|| ||W||
                pn = np.linalg.norm(H, axis=(1, 2)) * np.linalg.norm(Phi, axis=(1, 2)) * np.linalg.norm(W, axis=(1, 2))
                bad = pn > cP * i ** nu
                if bad.any():
                    step[bad] = 0.0; n_trunc += bad.sum()
            bn = beta - gam * step
            if box is not None:
                cl = np.clip(bn, -box, box); n_proj += (cl != bn).any(1).sum(); bn = cl
            # ---- updates of Phi, W, H at the averaged iterate
            r_b, a_vec = model.r(x, y, bbar)
            gb = z * r_b[:, None]
            s = z if method == "s2sls" else gb
            Ws = np.einsum("rij,rj->ri", W, s); m = N + (s * Ws).sum(1)
            if method != "first":
                Wz = Ws if method == "s2sls" else np.einsum("rij,rj->ri", W, z)
                p = np.einsum("rki,rk->ri", Phi, Wz); qv = np.einsum("rki,rk->ri", Phi, Ws)
                czz = (z * Wz).sum(1); czs = (z * Ws).sum(1)
                U = np.stack([p, qv, a_vec], 2)                                   # (R,d,3)
                C = np.zeros((R, 3, 3)); C[:, 0, 2] = C[:, 2, 0] = 1.0
                C[:, 1, 1] = -N / m; C[:, 1, 2] = C[:, 2, 1] = -czs / m; C[:, 2, 2] = (czz - czs ** 2 / m) / N
                HU = np.einsum("rij,rjk->rik", H, U)
                T = I3 + np.einsum("rij,rjk->rik", C, np.einsum("rji,rjk->rik", U, HU)) / N
                Xs = np.linalg.solve(T, np.einsum("rij,rkj->rik", C, HU))      # C U'H  (H symmetric)
                H = (N + 1) / N * (H - np.einsum("rij,rjk->rik", HU, Xs) / N)
                H = 0.5 * (H + np.transpose(H, (0, 2, 1)))
            Phi = (N * Phi + z[:, :, None] * a_vec[:, None, :]) / (N + 1)
            W = (N + 1) / N * (W - Ws[:, :, None] * Ws[:, None, :] / m[:, None, None])
            # ---- accumulators for J and sandwich (all evaluated at bbar_{i-1})
            Sg_acc += gb
            SGb += z * (a_vec * bbar).sum(1)[:, None]
            SG += z[:, :, None] * a_vec[:, None, :]
            if method == "s2sls":
                Om += gb[:, :, None] * gb[:, None, :]
            beta = bn
            bbar = bbar + (beta - bbar) / i
            w2 = float(i) ** 2; A0 += w2; A1 += w2 * bbar; A2 += w2 * bbar[:, :, None] * bbar[:, None, :]
            if i in checkpoints:
                n = i
                ghat = (Sg_acc - SGb + np.einsum("rij,rj->ri", SG, bbar)) / n
                J = n * np.einsum("ri,rij,rj->r", ghat, W, ghat)
                if method == "s2sls":
                    PW = np.einsum("rij,rkj,rkl->ril", H, Phi, W)
                    V = np.einsum("rij,rjk,rlk->ril", PW, Om / n, PW) / n
                elif method == "eff":
                    V = H / n
                else:
                    V = None
                VRS = (A2 - A1[:, :, None] * bbar[:, None, :] - bbar[:, :, None] * A1[:, None, :] + A0 * bbar[:, :, None] * bbar[:, None, :]) / n ** 2
                res[n] = dict(VRS=VRS.copy(), bbar=bbar.copy(), beta=beta.copy(), V=None if V is None else V.copy(), J=J,
                              time=time.perf_counter() - t0)
        if verbose: print(c, time.perf_counter() - t0, flush=True)
    res['n_proj'] = n_proj; res['n_trunc'] = n_trunc
    return res
