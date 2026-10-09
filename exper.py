import sys, json, time, numpy as np
from sgmm import *
stage = sys.argv[1]
CK = [10_000, 100_000, 1_000_000]
def summ(res, M):
    out = {}
    for n in CK:
        r = res[n]; err = r['bbar'] - M.beta; o = dict(bias=err.mean(0).tolist(), rmse=np.sqrt((err**2).mean(0)).tolist())
        if r['V'] is not None:
            se = np.sqrt(np.array([np.diag(v) for v in r['V']]))
            o['cover'] = (np.abs(err) <= 1.96 * se).mean(0).tolist(); o['len'] = (2 * 1.96 * se).mean(0).tolist()
        seR = np.sqrt(np.array([np.diag(v) for v in r['VRS']]) / n)
        o['cover_rs'] = (np.abs(err) <= 6.747 * seR).mean(0).tolist(); o['len_rs'] = (2 * 6.747 * seR).mean(0).tolist()
        o['J_rej'] = float((r['J'] > chi2q).mean()); o['J_mean'] = float(r['J'].mean())
        out[n] = o
    out['n_proj'] = int(res['n_proj']); out['n_trunc'] = int(res['n_trunc'])
    return out
from scipy import stats
if stage in ("lin_online", "exp_online", "expmis_online"):
    M = {"lin_online": LinearIV(), "exp_online": ExpIV(), "expmis_online": ExpIV(delta=0.01)}[stage]
    chi2q = stats.chi2.ppf(0.95, M.q - M.d)
    meths = ["eff"] if stage == "expmis_online" else ["eff", "s2sls", "first"]
    R = 200; out = {}
    for meth in meths:
        g0 = 1.0 if meth != "first" else float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
        t = time.perf_counter()
        res = run_online(M, 11, R, CK[-1], set(CK), method=meth, gamma0=g0, verbose=False)
        out[meth] = summ(res, M); out[meth]['wall'] = time.perf_counter() - t
        print(meth, json.dumps(out[meth][CK[-1]]), flush=True)
        np.save(f"bbar_{stage}_{meth}.npy", np.stack([res[n]['bbar'] for n in CK]))
    json.dump(out, open(f"res_{stage}.json", "w"), indent=1)
if stage in ("lin_off", "exp_off", "expmis_off"):
    M = {"lin_off": LinearIV(), "exp_off": ExpIV(), "expmis_off": ExpIV(delta=0.01)}[stage]
    chi2q = stats.chi2.ppf(0.95, M.q - M.d)
    R = 200; L = 10_000; n0 = 1000
    rec = {n: dict(b1=[], V1=[], b2=[], V2=[], J=[], t=[]) for n in CK}
    for r in range(R):
        z, x, y = chunk(M, 11, r, -1, n0); Zs = [z]; Xs = [x]; Ys = [y]
        for c in range(CK[-1] // L):
            z, x, y = chunk(M, 11, r, c, L); Zs.append(z); Xs.append(x); Ys.append(y)
        Z = np.vstack(Zs); X = np.vstack(Xs); Y = np.concatenate(Ys)
        for n in CK:
            t = time.perf_counter()
            b1, V1, b2, V2, J = offline_two_step(M, Z[:n0 + n], X[:n0 + n], Y[:n0 + n], np.zeros(M.d))
            rec[n]['t'].append(time.perf_counter() - t)
            for k, v in zip(['b1', 'V1', 'b2', 'V2', 'J'], [b1, V1, b2, V2, J]): rec[n][k].append(v)
        if r % 20 == 0: print(r, flush=True)
    out = {}
    for n in CK:
        o = {}
        for tag, bk, vk in [("2sls", "b1", "V1"), ("gmm", "b2", "V2")]:
            err = np.array(rec[n][bk]) - M.beta; se = np.sqrt(np.array([np.diag(v) for v in rec[n][vk]]))
            o[tag] = dict(bias=err.mean(0).tolist(), rmse=np.sqrt((err**2).mean(0)).tolist(),
                          cover=(abs(err) <= 1.96 * se).mean(0).tolist(), len=(2 * 1.96 * se).mean(0).tolist())
        J = np.array(rec[n]['J']); o['gmm']['J_rej'] = float((J > chi2q).mean()); o['gmm']['J_mean'] = float(J.mean())
        o['time'] = float(np.mean(rec[n]['t'])); out[n] = o
        np.save(f"b_{stage}_{n}.npy", np.array(rec[n]['b2']))
    json.dump(out, open(f"res_{stage}.json", "w"), indent=1)
    print(json.dumps(out[CK[-1]]))
if stage in ("rs_lin", "rs_exp"):
    M = LinearIV() if stage == "rs_lin" else ExpIV(); chi2q = stats.chi2.ppf(0.95, M.q - M.d)
    res = run_online(M, 11, 200, CK[-1], set(CK), method="eff", gamma0=1.0)
    out = summ(res, M); json.dump(out, open(f"res_{stage}.json", "w"), indent=1); print(json.dumps(out[CK[-1]]))
if stage == "sens":
    M = ExpIV(); chi2q = stats.chi2.ppf(0.95, M.q - M.d); n = 100_000; out = {}
    for meth in ["eff", "first"]:
        for g0 in [0.1, 0.3, 1.0, 3.0, 10.0]:
            res = run_online(M, 21, 100, n, {n}, method=meth, gamma0=g0)
            err = res[n]['bbar'] - M.beta
            out[f"{meth}_{g0}"] = dict(rmse=np.sqrt((err ** 2).mean(0)).tolist(), bias=err.mean(0).tolist(),
                                       n_proj=int(res['n_proj']))
            print(meth, g0, out[f"{meth}_{g0}"], flush=True)
    json.dump(out, open("res_sens.json", "w"), indent=1)
if stage == "expJ":
    M = ExpIV(); R = 500; CKJ = [10_000, 100_000]; out = {}
    res = run_online(M, 31, R, CKJ[-1], set(CKJ), method="eff", gamma0=1.0)
    offJ = {n: [] for n in CKJ}
    for r in range(R):
        z, x, y = chunk(M, 31, r, -1, 1000); Zs = [z]; Xs = [x]; Ys = [y]
        for c in range(CKJ[-1] // 10_000):
            z, x, y = chunk(M, 31, r, c, 10_000); Zs.append(z); Xs.append(x); Ys.append(y)
        Z = np.vstack(Zs); X = np.vstack(Xs); Y = np.concatenate(Ys)
        for n in CKJ:
            offJ[n].append(offline_two_step(M, Z[:1000 + n], X[:1000 + n], Y[:1000 + n], np.zeros(M.d))[4])
    for n in CKJ:
        out[n] = dict(online=res[n]['J'].tolist(), offline=offJ[n])
    json.dump(out, open("res_expJ.json", "w"))
    for n in CKJ:
        for k in ['online', 'offline']:
            J = np.array(out[n][k]); print(n, k, [float((J > stats.chi2.ppf(1 - al, 3)).mean()) for al in (0.1, 0.05, 0.01)])
