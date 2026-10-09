"""Builds the LaTeX tables and the figure of Section 6 from the JSON results."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("paper/gen", exist_ok=True)
CK = ["10000", "100000", "1000000"]
NL = {"10000": "$10^4$", "100000": "$10^5$", "1000000": "$10^6$"}
f3 = lambda x: f"{1e3 * x:.2f}"
fc = lambda x: f"{x:.3f}"[1:] if x < 1 else "1.00"


def acc_table(mdl, label, caption):
    on = json.load(open(f"res_{mdl}_online.json")); off = json.load(open(f"res_{mdl}_off.json"))
    rows = []
    for n in CK:
        meth = [("SGMM (efficient)", on["eff"][n], True, True), ("online 2SLS", on["s2sls"][n], True, True),
                ("first-order, $\\gamma_0=0.1$", on["first"][n], False, True),
                ("offline 2SLS", off[n]["2sls"], True, False), ("offline two-step GMM", off[n]["gmm"], True, False)]
        for k, (name, o, has_cov, has_rs) in enumerate(meth):
            r = " & ".join(f3(x) for x in o["rmse"])
            c = " & ".join(fc(x) for x in o["cover"]) if has_cov else "-- & -- & --"
            rs = fc(o["cover_rs"][1]) if has_rs else "--"
            rows.append(f"{NL[n] if k == 0 else ''} & {name} & {r} & {c} & {rs} \\\\")
        if n != CK[-1]: rows.append("\\midrule")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]\centering
\caption{{{caption}}}\label{{{label}}}
\footnotesize\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}cl ccc ccc c@{{}}}}\toprule
& & \multicolumn{{3}}{{c}}{{RMSE $\times10^3$}} & \multicolumn{{3}}{{c}}{{coverage, plug-in}} & RS\\ \cmidrule(lr){{3-5}}\cmidrule(lr){{6-8}}\cmidrule(l){{9-9}}
$n$ & method & $\beta_0$ & $\beta_1$ & $\beta_2$ & $\beta_0$ & $\beta_1$ & $\beta_2$ & $\beta_1$\\ \midrule
{body}
\bottomrule\end{{tabular}}\end{{table}}
"""


cap_lin = (r"Linear IV model (Example~\ref{ex:linear}), $\db=3$, $\dg=8$, $200$ replications. RMSE of the estimators, "
           r"empirical coverage of nominal $95\%$ plug-in confidence intervals (SGMM: $H_n/n$; online and offline 2SLS: sandwich), "
           r"and coverage of random-scaling (RS) intervals for the coefficient $\beta_1$ of the endogenous regressor $x_1$. Online methods use a pilot sample "
           r"of $n_0=1000$ and $\gamma_i=\gamma_0(i+100)^{-3/4}$ with $\gamma_0=1$ unless stated; offline methods use the same $n_0+n$ observations.")
cap_exp = (r"Exponential IV model (Example~\ref{ex:nonlinear}), $\db=3$, $\dg=6$, $200$ replications; columns as in Table~\ref{tab:lin}.")
open("paper/gen/tab_lin.tex", "w").write(acc_table("lin", "tab:lin", cap_lin))
open("paper/gen/tab_exp.tex", "w").write(acc_table("exp", "tab:exp", cap_exp))

# ---------------- J test
on_l = json.load(open("res_lin_online.json")); off_l = json.load(open("res_lin_off.json"))
on_e = json.load(open("res_exp_online.json")); off_e = json.load(open("res_exp_off.json"))
mis_on = json.load(open("res_expmis_online.json")); mis_off = json.load(open("res_expmis_off.json"))
rows = []
for n in CK:
    rows.append(f"{NL[n]} & {on_l['eff'][n]['J_rej']:.3f} & {off_l[n]['gmm']['J_rej']:.3f} & {on_e['eff'][n]['J_rej']:.3f} & {off_e[n]['gmm']['J_rej']:.3f} & {mis_on['eff'][n]['J_rej']:.3f} & {off_mis if False else mis_off[n]['gmm']['J_rej']:.3f} \\\\")
tabJ = r"""\begin{table}[t]\centering
\caption{Rejection rates of the online statistic $\mathcal J_n$ of \eqref{eq:Jn} and of the offline Hansen--Sargan statistic at the nominal $5\%$ level, $200$ replications. Size: correctly specified linear ($\chi^2_5$) and exponential ($\chi^2_3$) models. Power: exponential model in which the instrument $z_4$ is invalid ($\vartheta=0.01$).}\label{tab:J}
\small\begin{tabular}{@{}c cc cc cc@{}}\toprule
& \multicolumn{2}{c}{size, linear} & \multicolumn{2}{c}{size, exponential} & \multicolumn{2}{c}{power, exponential}\\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
$n$ & online & offline & online & offline & online & offline\\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n"
if os.path.exists("res_expJ.json"):
    from scipy import stats
    J = json.load(open("res_expJ.json")); r2 = []
    for n in ["10000", "100000"]:
        for k, name in [("online", "online $\\mathcal J_n$"), ("offline", "offline")]:
            x = np.array(J[n][k]); r2.append(f"{NL[n]} & {name} & " + " & ".join(f"{(x > stats.chi2.ppf(1 - a, 3)).mean():.3f}" for a in (0.10, 0.05, 0.01)) + f" & {x.mean():.2f} \\\\")
    tabJ += r"""
\begin{table}[t]\centering
\caption{Size of the overidentification tests in the exponential IV model with $500$ replications ($\chi^2_3$ limit, mean $3$).}\label{tab:J2}
\small\begin{tabular}{@{}cl ccc c@{}}\toprule
$n$ & statistic & $10\%$ & $5\%$ & $1\%$ & mean\\ \midrule
""" + "\n".join(r2) + "\n\\bottomrule\\end{tabular}\\end{table}\n"
open("paper/gen/tab_J.tex", "w").write(tabJ)

# ---------------- sensitivity
S = json.load(open("res_sens.json")); G = [0.1, 0.3, 1.0, 3.0, 10.0]
fmt = lambda v: f"{1e3 * v:.2f}" if v < 1 else f"{v:.2f}"
rows = []
for meth, name in [("eff", "SGMM (Gauss--Newton)"), ("first", "first-order")]:
    rows.append(name + " & " + " & ".join(fmt(S[f"{meth}_{g}"]["rmse"][1]) for g in G) + " \\\\")
    rows.append("\\quad projections & " + " & ".join(str(S[f"{meth}_{g}"]["n_proj"]) for g in G) + " \\\\")
tabS = r"""\begin{table}[t]\centering
\caption{Sensitivity to $\gamma_0$ in the exponential IV model, $n=10^5$, $100$ replications: RMSE$\,\times10^3$ of the averaged estimate of $\beta_1$ (values $\ge1$ are reported unscaled), and total number of steps at which the projection onto $[-10,10]^3$ was active.}\label{tab:sens}
\small\begin{tabular}{@{}l ccccc@{}}\toprule
$\gamma_0$ & $0.1$ & $0.3$ & $1$ & $3$ & $10$\\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n"
open("paper/gen/tab_sens.tex", "w").write(tabS)

# ---------------- figure: total RMSE versus n
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True, "grid.alpha": .3})
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
for a, (mdl, title) in zip(ax, [("lin", "(a) linear IV"), ("exp", "(b) exponential IV")]):
    on = json.load(open(f"res_{mdl}_online.json")); off = json.load(open(f"res_{mdl}_off.json"))
    n = np.array([int(c) for c in CK]); tot = lambda o: np.sqrt(np.sum(np.array(o["rmse"]) ** 2))
    a.loglog(n, [tot(on["eff"][c]) for c in CK], "o-", label="SGMM (efficient)")
    a.loglog(n, [tot(on["s2sls"][c]) for c in CK], "s-", label="online 2SLS")
    a.loglog(n, [tot(on["first"][c]) for c in CK], "^-", label=r"first-order, $\gamma_0=0.1$")
    a.loglog(n, [tot(off[c]["gmm"]) for c in CK], "k--", label="offline two-step GMM")
    a.loglog(n, [tot(off[c]["2sls"]) for c in CK], "k:", label="offline 2SLS")
    a.set_title(title); a.set_xlabel("$n$"); a.set_ylabel(r"RMSE of $\bar\beta_n$ (Euclidean)")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("paper/rmse.pdf"); plt.close(fig)

# ---------------- SV
cand = ["res_sv_1000000_0.75_1.0_1000_1.json"]
if cand:
    R = json.load(open(cand[0])); TH = np.array([-0.736, 0.90, 0.363]); rows = []
    for n in [c for c in CK if c in R]:
        off = np.array(R[n]["off"]) - TH; sg = np.array(R[n]["sgmm"]) - TH
        trapped = np.where(np.abs(np.array(R[n]["sgmm"])[:, 1] - 0.9) > 0.3)[0]
        keep = np.setdiff1d(np.arange(len(sg)), trapped)
        for k, (e, name) in enumerate([(off, "offline two-step GMM"), (sg, f"SGMM, all {len(sg)} replications"),
                                       (sg[keep], f"SGMM, without the trapped replication")]):
            rows.append(f"{NL[n] if k == 0 else ''} & {name} & " + " & ".join(f"{b:.4f}" for b in e.mean(0)) + " & " + " & ".join(f"{r:.4f}" for r in np.sqrt((e ** 2).mean(0))) + " \\\\")
        print("SV", n, "trapped", trapped, "MAE sgmm", np.median(np.abs(sg[keep]), 0), "MAE off", np.median(np.abs(off), 0))
        if n != CK[-1]: rows.append("\\midrule")
    nrep = len(R["pilot"])
    tabSV = r"""\begin{table}[t]\centering
\caption{Stochastic volatility model with the $24$ moments of Example~\ref{ex:sv}, $(\omega,b,\sigma_u)=(-0.736,0.90,0.363)$, NREP replications, pilot sample $n_0=5000$. Bias and RMSE of SGMM (per-observation updates, $\gamma_i=(i+1000)^{-3/4}$, parametrisation $(\mu,b,\log\tau^2)$) and of offline two-step GMM with the same (non-HAC) weight. The data are dependent, so this example is outside the scope of the theory.}\label{tab:sv}
\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}cl ccc ccc@{}}\toprule
& & \multicolumn{3}{c}{bias} & \multicolumn{3}{c}{RMSE}\\ \cmidrule(lr){3-5}\cmidrule(l){6-8}
$n$ & method & $\omega$ & $b$ & $\sigma_u$ & $\omega$ & $b$ & $\sigma_u$\\ \midrule
""".replace("NREP", str(nrep)) + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n"
    open("paper/gen/tab_sv.tex", "w").write(tabSV)
print("done")
