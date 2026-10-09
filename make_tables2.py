"""Tables and figure of Sections 7-8 (revised version) from the JSON results."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("paper/gen", exist_ok=True)
CK = ["10000", "100000", "1000000"]; NL = {"10000": "$10^4$", "100000": "$10^5$", "1000000": "$10^6$"}
f3 = lambda x: f"{1e3 * x:.2f}"
fc = lambda x: f"{x:.3f}"[1:] if x < 1 else "1.00"
ld = lambda f: json.load(open(f))


def acc_table(mname, label, caption):
    rec = ld(f"resc_main_{mname}.json"); rows = []
    for n in CK:
        meth = [("SGMM (efficient)", rec["eff"][n], True, True), ("online 2SLS", rec["s2sls"][n], True, True),
                ("first-order", rec["first"][n], False, True),
                ("offline 2SLS", rec["off2sls"][n], True, False), ("offline two-step GMM", rec["offgmm"][n], True, False)]
        for k, (name, o, hc, hr) in enumerate(meth):
            r = " & ".join(f3(x) for x in o["rmse"])
            c = " & ".join(fc(x) for x in o["cover"]) if hc else "-- & -- & --"
            rs = fc(o["cover_rs"][1]) if hr else "--"
            rows.append(f"{NL[n] if k == 0 else ''} & {name} & {r} & {c} & {rs} \\\\")
        if n != CK[-1]: rows.append("\\midrule")
    return rf"""\begin{{table}}[t]\centering
\caption{{{caption}}}\label{{{label}}}
\footnotesize\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}cl ccc ccc c@{{}}}}\toprule
& & \multicolumn{{3}}{{c}}{{RMSE $\times10^3$}} & \multicolumn{{3}}{{c}}{{coverage, plug-in}} & RS\\ \cmidrule(lr){{3-5}}\cmidrule(lr){{6-8}}\cmidrule(l){{9-9}}
$n$ & method & $\beta_0$ & $\beta_1$ & $\beta_2$ & $\beta_0$ & $\beta_1$ & $\beta_2$ & $\beta_1$\\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n"


cap_lin = (r"Linear IV model (Example~\ref{ex:linear}), $\db=3$, $\dg=8$, $1000$ replications. RMSE of the estimators, empirical coverage "
           r"of nominal $95\%$ plug-in confidence intervals (SGMM: $H_n/n$; online and offline 2SLS: sandwich), and coverage of random-scaling (RS) "
           r"intervals for the coefficient $\beta_1$ of the endogenous regressor $x_1$. Online methods use a pilot sample of $n_0=1000$ and the default "
           r"steps of Remark~\ref{rem:design}(v); offline methods use the same $n_0+n$ observations. The Monte Carlo standard error of a coverage of "
           r"$0.95$ is $0.007$.")
cap_exp = r"Exponential IV model (Example~\ref{ex:nonlinear}), $\db=3$, $\dg=6$, $1000$ replications; columns as in Table~\ref{tab:lin}."
open("paper/gen/tab_lin.tex", "w").write(acc_table("linear", "tab:lin", cap_lin))
open("paper/gen/tab_exp.tex", "w").write(acc_table("exp", "tab:exp", cap_exp))

# ---------------- J tests
jf = ld("resc_jfix.json"); ml = ld("resc_main_linear.json"); me = ld("resc_main_exp.json"); mis = ld("resc_mis_exp.json")
rows = []
for mname, rec, lab in [("linear", ml, "linear ($\\chi^2_5$)"), ("exp", me, "exponential ($\\chi^2_3$)")]:
    for k, n in enumerate(CK):
        on = jf[mname][n]["Jp_rej"]; onu = jf[mname][n]["J_rej"]; off = rec["offgmm"][n]["J_rej"]
        rows.append(f"{lab if k == 0 else ''} & {NL[n]} & " + " & ".join(f"{on[a]:.3f}" for a in ("0.1", "0.05", "0.01")) + " & "
                    + " & ".join(f"{off[a]:.3f}" for a in ("0.1", "0.05", "0.01")) + f" & {onu['0.05']:.3f} \\\\")
    rows.append("\\midrule")
for k, n in enumerate(CK):
    on = mis["eff"][n]["Jp_rej"]; off = mis["offgmm"][n]["J_rej"]; onu = mis["eff"][n]["J_rej"]
    rows.append(f"{'power, $\\vartheta=0.01$' if k == 0 else ''} & {NL[n]} & " + " & ".join(f"{on[a]:.3f}" for a in ("0.1", "0.05", "0.01")) + " & "
                + " & ".join(f"{off[a]:.3f}" for a in ("0.1", "0.05", "0.01")) + f" & {onu['0.05']:.3f} \\\\")
tabJ = r"""\begin{table}[t]\centering
\caption{Rejection rates of the online statistic $\mathcal J_n$ of \eqref{eq:Jn} and of the offline Hansen--Sargan statistic at the nominal $10\%$, $5\%$ and $1\%$ levels, $1000$ replications. Size: correctly specified linear and exponential models. Power: exponential model in which the instrument $z_4$ is invalid ($\vartheta=0.01$). Last column: unprojected online statistic $n\hat g_n\tr W_n\hat g_n$ at the $5\%$ level.}\label{tab:J}
\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}lc ccc ccc c@{}}\toprule
& & \multicolumn{3}{c}{online $\mathcal J_n$} & \multicolumn{3}{c}{offline} & unproj.\\ \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-9}
& $n$ & $10\%$ & $5\%$ & $1\%$ & $10\%$ & $5\%$ & $1\%$ & $5\%$\\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n"
open("paper/gen/tab_J.tex", "w").write(tabJ)

# ---------------- sensitivity / units
S = ld("resc_sens.json"); C = [0.5, 1.0, 2.0, 4.0, 8.0]
fmt = lambda v: f"{1e3 * v:.2f}" if v < 1 else f"{v:.1f}"
rows = []
for tag, tname in [("orig", "original units"), ("scaled", "$x_2$ rescaled")]:
    for meth, name in [("eff", "SGMM"), ("first", "first-order")]:
        rows.append(f"{tname if meth == 'eff' else ''} & {name} & " + " & ".join(fmt(S[f'{tag}_{meth}_{c}']['rmse'][1]) for c in C) + " \\\\")
tabS = r"""\begin{table}[t]\centering
\caption{Exponential IV model, $n=10^5$, $200$ replications: RMSE$\,\times10^3$ of the averaged estimate of $\beta_1$ (values $\ge1$ unscaled) for steps $\gamma_i=c_\gamma n_0^{a-1}(n_0+i)^{-a}$ and several $c_\gamma$, in the original units and after expressing $x_2$ (regressor and instrument) in units ten times smaller.}\label{tab:sens}
\small\begin{tabular}{@{}ll ccccc@{}}\toprule
& $c_\gamma$ & $0.5$ & $1$ & $2$ & $4$ & $8$\\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n"
open("paper/gen/tab_sens.tex", "w").write(tabS)

# ---------------- timing
T = ld("res_timing.json"); TS = ld("res_timing_stream.json"); rows = []
for n in ["100000", "1000000", "10000000"]:
    t = T[n]; rows.append(f"$10^{{{len(n) - 1}}}$ & {t['t_online']:.3f} & {t['t_offline']:.3f} & {t['per_obs_us']:.2f} & {t['mem_data_MB']:.0f} \\\\")
tabT = r"""\begin{table}[t]\centering
\caption{Computing time (seconds, best of three) and memory in the exponential IV model: single-pass SGMM in C (one thread) versus offline two-step GMM in NumPy with all data in memory (multithreaded linear algebra), on the same two-core machine. SGMM stores $O(\dg^2+\dg\db)$ numbers (less than $1$ KB here).}\label{tab:time}
\small\begin{tabular}{@{}c cc c c@{}}\toprule
$n$ & SGMM & offline GMM & SGMM, $\mu$s per observation & data in memory (MB)\\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n"
open("paper/gen/tab_time.tex", "w").write(tabT)
print("stream", TS["online_total"], TS["offline_total"])

# ---------------- application
A = ld("res_application.json"); names = ["constant", "morekids", "boy first", "age", "age$^2$", "black", "hispanic", "other race"]
B = np.array([r["b"] for r in A["sgmm"]]); SE = np.array([r["se"] for r in A["sgmm"]]); J = np.array([r["J"] for r in A["sgmm"]])
rows = []
for k, nm in enumerate(names):
    rows.append(f"{nm} & {A['offline']['b'][k]:.4f} & ({A['offline']['se'][k]:.4f}) & {A['sgmm'][0]['b'][k]:.4f} & ({A['sgmm'][0]['se'][k]:.4f}) & {B[:, k].mean():.4f} & {B[:, k].std():.4f} \\\\")
tabA = r"""\begin{table}[t]\centering
\caption{Weeks worked and a third child: exponential IV model estimated by offline two-step GMM and by single-pass SGMM ($n=254{,}654$; pilot $n_0=5000$). Columns 4--5: SGMM for one random order of the observations, with plug-in standard errors; columns 6--7: mean and standard deviation of the SGMM estimates over $20$ random orders.}\label{tab:app}
\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}l cc cc cc@{}}\toprule
& \multicolumn{2}{c}{offline GMM} & \multicolumn{2}{c}{SGMM, one order} & \multicolumn{2}{c}{SGMM, 20 orders}\\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
& estimate & (s.e.) & estimate & (s.e.) & mean & s.d.\\ \midrule
""" + "\n".join(rows) + f"""
\\midrule $J$ statistic ($\\chi^2_1$) & {A['offline']['J']:.2f} & & {A['sgmm'][0]['J']:.2f} & & {J.mean():.2f} & {J.std():.2f} \\\\
time (s) & {A['offline']['time']:.2f} & & {A['sgmm'][0]['time']:.2f} & & & \\\\
\\bottomrule\\end{{tabular}}\\end{{table}}
"""
open("paper/gen/tab_app.tex", "w").write(tabA)

# ---------------- SV
def sv_rows(f, tag_off, tag_on):
    R = ld(f); TH = np.array([-0.736, 0.90, 0.363]); rows = []
    for n in [c for c in CK if c in R]:
        for k, (key, name) in enumerate([("off", tag_off), ("sgmm", tag_on)]):
            e = np.array(R[n][key]) - TH
            rows.append(f"{NL[n] if k == 0 else ''} & {name} & " + " & ".join(f"{b:.4f}" for b in e.mean(0)) + " & " + " & ".join(f"{r:.4f}" for r in np.sqrt((e ** 2).mean(0))) + " \\\\")
        if n != CK[-1]: rows.append("\\cmidrule(l){2-8}")
    return rows, len(R["pilot"])
parts = []
for f, lab, to, tn in [("res_svdep_1000000_D0_L0_c2.0.json", "(a) weights from $\\E[gg\\tr]$", "offline GMM", "SGMM"),
                        ("res_svdep_1000000_D0_L50_c2.0.json", "(b) Bartlett long-run weights, $L=50$", "offline HAC GMM", "SGMM, long-run weights")]:
    if os.path.exists(f):
        rows, nrep = sv_rows(f, to, tn); parts.append(f"\\multicolumn{{8}}{{@{{}}l}}{{{lab}}}\\\\ \\midrule\n" + "\n".join(rows))
if parts:
    tabSV = r"""\begin{table}[t]\centering
\caption{Stochastic volatility model with the $24$ moments of Example~\ref{ex:sv}, $(\omega,b,\sigma_u)=(-0.736,0.90,0.363)$, NREP replications, pilot $n_0=5000$, default steps; computations in the parametrisation $(\mu,b,\log\tau^2)$. Bias and RMSE of SGMM and of offline two-step GMM with the same type of weight.}\label{tab:sv}
\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}cl ccc ccc@{}}\toprule
& & \multicolumn{3}{c}{bias} & \multicolumn{3}{c}{RMSE}\\ \cmidrule(lr){3-5}\cmidrule(l){6-8}
$n$ & method & $\omega$ & $b$ & $\sigma_u$ & $\omega$ & $b$ & $\sigma_u$\\ \midrule
""".replace("NREP", str(nrep)) + "\n\\midrule\n".join(parts) + "\n\\bottomrule\\end{tabular}\\end{table}\n"
    open("paper/gen/tab_sv.tex", "w").write(tabSV)

# ---------------- figure
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True, "grid.alpha": .3})
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
for a_, (mname, title) in zip(ax, [("linear", "(a) linear IV"), ("exp", "(b) exponential IV")]):
    rec = ld(f"resc_main_{mname}.json"); n = np.array([int(c) for c in CK]); tot = lambda o: np.sqrt(np.sum(np.array(o["rmse"]) ** 2))
    a_.loglog(n, [tot(rec["eff"][c]) for c in CK], "o-", label="SGMM (efficient)")
    a_.loglog(n, [tot(rec["s2sls"][c]) for c in CK], "s-", label="online 2SLS")
    a_.loglog(n, [tot(rec["first"][c]) for c in CK], "^-", label="first-order")
    a_.loglog(n, [tot(rec["offgmm"][c]) for c in CK], "k--", label="offline two-step GMM")
    a_.loglog(n, [tot(rec["off2sls"][c]) for c in CK], "k:", label="offline 2SLS")
    a_.set_title(title); a_.set_xlabel("$n$"); a_.set_ylabel(r"RMSE of $\bar\beta_n$ (Euclidean)")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("paper/rmse.pdf"); plt.close(fig)
print("done")
