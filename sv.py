"""Log-normal stochastic volatility model with the 24 moments of Jacquier, Polson and Rossi /
Andersen and Sorensen: online SGMM versus offline two-step GMM (non-HAC weights)."""
import sys, json, time, numpy as np
from scipy.signal import lfilter

TH = np.array([-0.736, 0.90, 0.363])           # (omega, beta, sigma_u)
LAGS = np.arange(1, 11); C = 100.0             # moments of 100*y (percentage returns)
POW = np.array([1, 2, 3, 4] + [2] * 10 + [4] * 10)
SC = C ** POW


def model_moments(th):
    om, be, su = th
    mu = om / (1 - be); t2 = su ** 2 / (1 - be ** 2)
    Es = lambda r: np.exp(r * mu / 2 + r * r * t2 / 8)
    bj = be ** LAGS
    m = np.concatenate([[np.sqrt(2 / np.pi) * Es(1), Es(2), 2 * np.sqrt(2 / np.pi) * Es(3), 3 * Es(4)],
                        2 / np.pi * Es(1) ** 2 * np.exp(bj * t2 / 4), Es(2) ** 2 * np.exp(bj * t2)])
    # gradient: log m is linear in (mu, t2, bj*t2)
    dmu = np.array([1 / (1 - be), om / (1 - be) ** 2, 0.0])
    dt2 = np.array([0.0, 2 * be * su ** 2 / (1 - be ** 2) ** 2, 2 * su / (1 - be ** 2)])
    rr = np.array([1, 2, 3, 4]); dl = np.zeros((24, 3))
    dl[:4] = np.outer(rr / 2, dmu) + np.outer(rr ** 2 / 8, dt2)
    for k, (r, c) in enumerate([(1, 0.25), (2, 1.0)]):
        sl = slice(4 + 10 * k, 14 + 10 * k)
        dbt = np.outer(bj, dt2); dbt[:, 1] += LAGS * be ** (LAGS - 1) * t2
        dl[sl] = 2 * (np.outer(np.full(10, r / 2), dmu) + np.outer(np.full(10, r * r / 8), dt2)) + c * dbt
    return m * SC, (m * SC)[:, None] * dl


def features(y):
    """x_i for i >= 10 from a path y (length n+10) -> (n, 24), already scaled."""
    a = np.abs(y); n = len(y) - 10; cur = slice(10, None)
    f = [a[cur], y[cur] ** 2, a[cur] ** 3, y[cur] ** 4]
    f += [a[cur] * a[10 - j:len(y) - j] for j in LAGS]
    f += [y[cur] ** 2 * y[10 - j:len(y) - j] ** 2 for j in LAGS]
    return np.column_stack(f) * SC


class Stream:
    """Continuing simulation of R independent SV paths."""
    def __init__(self, R, seed):
        self.rng = np.random.default_rng(seed); self.R = R
        om, be, su = TH; mu = om / (1 - be); t2 = su ** 2 / (1 - be ** 2)
        self.h = mu + np.sqrt(t2) * self.rng.standard_normal(R); self.ytail = None

    def next(self, L):
        om, be, su = TH; R = self.R; out = np.empty((L, R, 24))
        for r in range(R):
            e = om + su * self.rng.standard_normal(L + (10 if self.ytail is None else 0))
            h, _ = lfilter([1.0], [1.0, -be], e, zi=[be * self.h[r]])
            self.h[r] = h[-1]
            y = np.exp(h / 2) * self.rng.standard_normal(len(h))
            yy = y if self.ytail is None else np.concatenate([self.ytail[r], y])
            if self.ytail is None: self._tmp = getattr(self, "_tmp", []); self._tmp.append(yy[-10:])
            else: self.ytail[r] = yy[-10:]
            out[:, r] = features(yy)
        if self.ytail is None: self.ytail = np.array(self._tmp)
        return out


def offline(xbar, S, th0, iters=100):
    W = np.linalg.inv(S); th = th0.copy()
    f = lambda th: (xbar - model_moments(th)[0]) @ W @ (xbar - model_moments(th)[0])
    fv = f(th)
    for _ in range(iters):
        m, D = model_moments(th); g = xbar - m; G = -D
        step = np.linalg.solve(G.T @ W @ G, G.T @ W @ g); t = 1.0
        while True:
            tn = np.clip(th - t * step, LO, HI); fn = f(tn)
            if fn <= fv or t < 1e-6: break
            t /= 2
        conv = abs(fv - fn) < 1e-14 * max(fv, 1e-300)
        th, fv = tn, fn
        if conv: break
    m, D = model_moments(th)
    return th, np.linalg.inv(D.T @ W @ D)


LO = np.array([-5.0, 0.3, 0.05]); HI = np.array([0.0, 0.99, 1.5])



def mm_vec(th):
    """Vectorised model_moments for th of shape (R,3): returns (R,24), (R,24,3)."""
    om, be, su = th[:, 0], th[:, 1], th[:, 2]
    mu = om / (1 - be); t2 = su ** 2 / (1 - be ** 2)
    Es = lambda r: np.exp(r * mu / 2 + r * r * t2 / 8)
    bj = be[:, None] ** LAGS
    m = np.concatenate([np.stack([np.sqrt(2 / np.pi) * Es(1), Es(2), 2 * np.sqrt(2 / np.pi) * Es(3), 3 * Es(4)], 1),
                        2 / np.pi * Es(1)[:, None] ** 2 * np.exp(bj * t2[:, None] / 4),
                        Es(2)[:, None] ** 2 * np.exp(bj * t2[:, None])], 1)
    Z = np.zeros_like(om)
    dmu = np.stack([1 / (1 - be), om / (1 - be) ** 2, Z], 1)
    dt2 = np.stack([Z, 2 * be * su ** 2 / (1 - be ** 2) ** 2, 2 * su / (1 - be ** 2)], 1)
    rr = np.array([1, 2, 3, 4.0]); dl = np.zeros((len(om), 24, 3))
    dl[:, :4] = (rr / 2)[None, :, None] * dmu[:, None, :] + (rr ** 2 / 8)[None, :, None] * dt2[:, None, :]
    for k, (r, c) in enumerate([(1, 0.25), (2, 1.0)]):
        dbt = bj[:, :, None] * dt2[:, None, :]
        dbt[:, :, 1] += LAGS * be[:, None] ** (LAGS - 1) * t2[:, None]
        dl[:, 4 + 10 * k:14 + 10 * k] = 2 * ((r / 2) * dmu[:, None, :] + (r * r / 8) * dt2[:, None, :]) + c * dbt
    M = m * SC
    return M, M[:, :, None] * dl


def mm_rep(ph):
    """Moments in the parametrisation ph = (mu, beta, nu), mu = omega/(1-beta), nu = log tau^2."""
    mu, be, nu = ph[:, 0], ph[:, 1], ph[:, 2]; t2 = np.exp(nu)
    Es = lambda r: np.exp(r * mu / 2 + r * r * t2 / 8)
    bj = be[:, None] ** LAGS
    m = np.concatenate([np.stack([np.sqrt(2 / np.pi) * Es(1), Es(2), 2 * np.sqrt(2 / np.pi) * Es(3), 3 * Es(4)], 1),
                        2 / np.pi * Es(1)[:, None] ** 2 * np.exp(bj * t2[:, None] / 4),
                        Es(2)[:, None] ** 2 * np.exp(bj * t2[:, None])], 1)
    R = len(mu); dl = np.zeros((R, 24, 3)); rr = np.array([1, 2, 3, 4.0])
    dl[:, :4, 0] = rr / 2; dl[:, :4, 2] = (rr ** 2 / 8)[None] * t2[:, None]
    for k, (r, c) in enumerate([(1, 0.25), (2, 1.0)]):
        sl = slice(4 + 10 * k, 14 + 10 * k)
        dl[:, sl, 0] = r
        dl[:, sl, 1] = c * LAGS * be[:, None] ** (LAGS - 1) * t2[:, None]
        dl[:, sl, 2] = (r * r / 4) * t2[:, None] + c * bj * t2[:, None]
    M = m * SC
    return M, M[:, :, None] * dl


def to_rep(th):
    om, be, su = th[..., 0], th[..., 1], th[..., 2]
    return np.stack([om / (1 - be), be, np.log(su ** 2 / (1 - be ** 2))], -1)


def from_rep(ph):
    mu, be, nu = ph[..., 0], ph[..., 1], ph[..., 2]
    return np.stack([mu * (1 - be), be, np.sqrt(np.exp(nu) * (1 - be ** 2))], -1)


def offline_rep(xbar, S, ph0, iters=200):
    W = np.linalg.inv(S); ph = ph0.copy()
    def f(p):
        g = xbar - mm_rep(p[None])[0][0]; return g @ W @ g
    fv = f(ph)
    for _ in range(iters):
        m, D = mm_rep(ph[None]); g = xbar - m[0]; G = -D[0]
        step = np.linalg.solve(G.T @ W @ G, G.T @ W @ g); t = 1.0
        while True:
            pn = np.clip(ph - t * step, PLO, PHI); fn = f(pn)
            if fn <= fv or t < 1e-6: break
            t /= 2
        conv = abs(fv - fn) < 1e-14 * max(fv, 1e-300)
        ph, fv = pn, fn
        if conv: break
    return ph


PLO = np.array([-12.0, 0.3, -4.0]); PHI = np.array([-3.0, 0.99, 1.0])

if __name__ == "__main__":
    R = int(sys.argv[1]); n_max = int(sys.argv[2]); n0 = 5000; L = 10000
    a, gamma0, i0 = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]); B = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    CK = [n for n in [10_000, 100_000, 1_000_000] if n <= n_max]
    st = Stream(R, 2024)
    X0 = st.next(n0)
    xbar0 = X0.mean(0); S0 = np.einsum("nri,nrj->rij", X0 - xbar0, X0 - xbar0) / n0
    p0 = to_rep(np.array([-1.0, 0.8, 0.3]))
    b0 = np.array([offline_rep(xbar0[r], S0[r], p0) for r in range(R)])
    beta = b0.copy(); bbar = b0.copy()
    M0, D0 = mm_rep(b0); Phi = -D0
    W = np.linalg.inv(np.einsum("nri,nrj->rij", X0 - M0, X0 - M0) / n0)
    sx = X0.sum(0); sxx = np.einsum("nri,nrj->rij", X0, X0)
    res = {}; t0 = time.perf_counter(); n_proj = 0; bw = np.zeros_like(b0); sw = 0.0
    for c in range(n_max // L):
        Xc = st.next(L)
        sx += Xc.sum(0); sxx += np.einsum("nri,nrj->rij", Xc, Xc)
        for t in range(L // B):
            i = (c * L) // B + t + 1; N = n0 / B + i - 1.0; x = Xc[t * B:(t + 1) * B].mean(0)
            gam = gamma0 * (i + i0) ** (-a)
            g = x - mm_rep(beta)[0]
            Hm = np.linalg.inv(np.einsum("rki,rkl,rlj->rij", Phi, W, Phi))
            step = np.einsum("rij,rkj,rkl,rl->ri", Hm, Phi, W, g)
            bn = beta - gam * step; cl = np.clip(bn, PLO, PHI); n_proj += (cl != bn).any(1).sum(); beta = cl
            mb, Db = mm_rep(bbar); s = x - mb
            sB = s * np.sqrt(B)   # W estimates the inverse covariance of single observations
            Ws = np.einsum("rij,rj->ri", W, sB); m = N + (sB * Ws).sum(1)
            W = (N + 1) / N * (W - Ws[:, :, None] * Ws[:, None, :] / m[:, None, None])
            Phi = (N * Phi - Db) / (N + 1)
            bbar = bbar + (beta - bbar) / i
            w = np.log(i + 1.0) ** 2; sw += w; bw += w * (beta - bw) / sw
        n = (c + 1) * L
        if n in CK:
            xb = sx / (n0 + n); Sx = sxx / (n0 + n) - np.einsum("ri,rj->rij", xb, xb)
            off = np.array([offline_rep(xb[r], Sx[r], b0[r]) for r in range(R)])
            H = np.linalg.inv(np.einsum("rki,rkl,rlj->rij", Phi, W, Phi))
            res[n] = dict(sgmm=from_rep(bbar).tolist(), sgmm_rep=bbar.tolist(), off=from_rep(off).tolist(),
                          off_rep=off.tolist(), H=H.tolist(), time=time.perf_counter() - t0)
            e3 = from_rep(bw) - TH; res[n]['wavg'] = from_rep(bw).tolist()
            e1 = from_rep(bbar) - TH; e2 = from_rep(off) - TH
            print(n, "SGMM rmse", np.round(np.sqrt((e1 ** 2).mean(0)), 4), "bias", np.round(e1.mean(0), 4), "| GMM rmse", np.round(np.sqrt((e2 ** 2).mean(0)), 4), "bias", np.round(e2.mean(0), 4), "proj", n_proj, "| wavg rmse", np.round(np.sqrt((e3 ** 2).mean(0)), 4), "bias", np.round(e3.mean(0), 4), "%.0fs" % (time.perf_counter() - t0), flush=True)
    res['pilot'] = from_rep(b0).tolist(); res['n_proj'] = int(n_proj); res['cfg'] = [a, gamma0, i0]
    json.dump(res, open(f"res_sv_{n_max}_{a}_{gamma0}_{i0}_{B}.json", "w"))
