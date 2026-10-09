"""Simulation experiments of Section 6 with the compiled engine.

usage: python3 exper_c.py main linear|exp      (R = 1000: online SGMM / online 2SLS / first-order, offline 2SLS / GMM)
       python3 exper_c.py mis                  (exponential model with an invalid instrument, theta = 0.01)
       python3 exper_c.py sens                 (step-size constant and units of measurement)
Default step: gamma_i = c n0^(a-1) (n0 + i)^(-a), c = 2, a = 3/4, n0 = 1000.
"""
import sys, json, time
import numpy as np
from scipy import stats
import sgmm
from engine import Engine, pilot_one

CK = [10_000, 100_000, 1_000_000]; L = 10_000; N0 = 1000; A = 0.75
G0 = lambda c: c * N0 ** (A - 1)


def online(M, mname, meth, seed, r, c=2.0, ck=CK):
    b0, G, W0 = pilot_one(M, seed, r, N0, meth)
    e = Engine(mname, meth, b0, G, W0, N0, G0(c), A, N0)
    out = {}
    for ci in range(ck[-1] // L):
        z, x, y = sgmm.chunk(M, seed, r, ci, L); e.run(z, x, y)
        if e.i in ck:
            s = e.summary(); out[e.i] = s
    out["nproj"] = e.nproj; out["ntrunc"] = e.ntrunc.value
    return out


def offline(M, seed, r, ck=CK):
    z, x, y = sgmm.chunk(M, seed, r, -1, N0); Zs = [z]; Xs = [x]; Ys = [y]
    for ci in range(ck[-1] // L):
        z, x, y = sgmm.chunk(M, seed, r, ci, L); Zs.append(z); Xs.append(x); Ys.append(y)
    Z = np.vstack(Zs); X = np.vstack(Xs); Y = np.concatenate(Ys); out = {}
    for n in ck:
        b1, V1, b2, V2, J = sgmm.offline_two_step(M, Z[:N0 + n], X[:N0 + n], Y[:N0 + n], np.zeros(M.d))
        out[n] = dict(b1=b1, V1=V1, b2=b2, V2=V2, J=J)
    return out


def summarize(errs, ses, ses_rs, J, q, Jp=None):
    errs = np.array(errs); o = dict(bias=errs.mean(0).tolist(), rmse=np.sqrt((errs ** 2).mean(0)).tolist())
    if ses is not None:
        se = np.array(ses); o["cover"] = (abs(errs) <= 1.96 * se).mean(0).tolist(); o["len"] = (2 * 1.96 * se).mean(0).tolist()
    if ses_rs is not None:
        se = np.array(ses_rs); o["cover_rs"] = (abs(errs) <= 6.747 * se).mean(0).tolist()
    if J is not None:
        J = np.array(J); o["J"] = J.tolist(); o["J_rej"] = {str(al): float((J > stats.chi2.ppf(1 - al, q)).mean()) for al in (0.1, 0.05, 0.01)}
        o["J_mean"] = float(J.mean())
    if Jp is not None:
        Jp = np.array(Jp); o["Jp"] = Jp.tolist(); o["Jp_rej"] = {str(al): float((Jp > stats.chi2.ppf(1 - al, q)).mean()) for al in (0.1, 0.05, 0.01)}
        o["Jp_mean"] = float(Jp.mean())
    return o


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage in ("main", "mis"):
        if stage == "main":
            mname = sys.argv[2]; M = sgmm.LinearIV() if mname == "linear" else sgmm.ExpIV(); R = 1000; seed = 12
            meths = ["eff", "s2sls", "first"]
        else:
            mname = "exp"; M = sgmm.ExpIV(delta=0.01); R = 1000; seed = 13; meths = ["eff"]
        q = M.q - M.d; rec = {}; t0 = time.perf_counter()
        for meth in meths:
            res = [online(M, mname, meth, seed, r) for r in range(R)]
            rec[meth] = {}
            for n in CK:
                errs = [o[n]["bbar"] - M.beta for o in res]
                ses = None if meth == "first" else [np.sqrt(np.diag(o[n]["V"])) for o in res]
                rss = [np.sqrt(np.diag(o[n]["VRS"])) for o in res]
                rec[meth][n] = summarize(errs, ses, rss, [o[n]["J"] for o in res] if meth == "eff" else None, q,
                                         [o[n]["Jp"] for o in res] if meth == "eff" else None)
            rec[meth]["nproj"] = int(sum(o["nproj"] for o in res)); rec[meth]["ntrunc"] = int(sum(o["ntrunc"] for o in res))
            print(meth, "done", time.perf_counter() - t0, json.dumps({k: v for k, v in rec[meth][CK[-1]].items() if k != "J"}), flush=True)
        offr = [offline(M, seed, r) for r in range(R)]
        for tag, bk, vk in [("off2sls", "b1", "V1"), ("offgmm", "b2", "V2")]:
            rec[tag] = {}
            for n in CK:
                rec[tag][n] = summarize([o[n][bk] - M.beta for o in offr], [np.sqrt(np.diag(o[n][vk])) for o in offr], None,
                                        [o[n]["J"] for o in offr] if tag == "offgmm" else None, q)
        print("offline done", time.perf_counter() - t0, flush=True)
        json.dump(rec, open(f"resc_{stage}_{mname}.json", "w"))
    if stage == "jfix":
        out = {}
        for mname, M in [("linear", sgmm.LinearIV()), ("exp", sgmm.ExpIV())]:
            res = [online(M, mname, "eff", 12, r) for r in range(1000)]
            out[mname] = {n: summarize([o[n]["bbar"] - M.beta for o in res], None, None, [o[n]["J"] for o in res], M.q - M.d,
                                       [o[n]["Jp"] for o in res]) for n in CK}
            print(mname, {n: (out[mname][n]["J_rej"]["0.05"], out[mname][n]["Jp_rej"]["0.05"]) for n in CK}, flush=True)
        json.dump(out, open("resc_jfix.json", "w"))
    if stage == "sens":
        class ExpScaled(sgmm.ExpIV):
            """Same data as ExpIV, with x2 (regressor and instrument) expressed in units ten times smaller."""
            beta = np.array([0.5, 0.5, -0.05])
            def gen(self, rng, n):
                z, x, y = sgmm.ExpIV().gen(rng, n); x = x.copy(); z = z.copy(); x[:, 2] *= 10; z[:, 5] *= 10; return z, x, y
        out = {}; R = 200; n = 100_000
        for tag, M in [("orig", sgmm.ExpIV()), ("scaled", ExpScaled())]:
            for meth in ["eff", "first"]:
                for c in [0.5, 1.0, 2.0, 4.0, 8.0]:
                    res = [online(M, "exp", meth, 21, r, c=c, ck=[n]) for r in range(R)]
                    errs = np.array([o[n]["bbar"] - M.beta for o in res])
                    out[f"{tag}_{meth}_{c}"] = dict(rmse=np.sqrt((errs ** 2).mean(0)).tolist(), nproj=int(sum(o["nproj"] for o in res)))
                    print(tag, meth, c, out[f"{tag}_{meth}_{c}"], flush=True)
        json.dump(out, open("resc_sens.json", "w"), indent=1)
