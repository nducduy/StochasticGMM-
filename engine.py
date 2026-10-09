"""Python driver for the compiled engine (sgmm_engine.c)."""
import ctypes, os, subprocess
import numpy as np
from sgmm import pilot

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sgmm_engine.c"); LIB = os.path.join(HERE, "libsgmm_engine.so")
if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
    subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", LIB, SRC, "-lm"])
_lib = ctypes.CDLL(LIB)
_P = ctypes.POINTER(ctypes.c_double); _L = ctypes.POINTER(ctypes.c_long)
_lib.sgmm_run.restype = ctypes.c_long
_lib.sgmm_run.argtypes = ([ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int, _P, _P, _P,
                           ctypes.c_long] + [ctypes.c_double] * 6 + [_P] * 10 + [_L])
_p = lambda a: a.ctypes.data_as(_P)
MODELS = {"linear": 0, "exp": 1}; METHODS = {"eff": 0, "s2sls": 1, "first": 2}


class Engine:
    """State of one replication of Algorithm 1 (or of a variant) driven by the C engine."""

    def __init__(self, model_name, method, b0, Phi0, W0, n0, gamma0, a, i0, box=10.0, cP=1e8, nu=0.1):
        self.model = MODELS[model_name]; self.method = METHODS[method]
        self.dg, self.db = Phi0.shape; self.n0 = n0; self.i = 0
        self.prm = (gamma0, a, float(i0), box, cP, nu)
        c = lambda v: np.ascontiguousarray(np.array(v, dtype=float))
        self.beta = c(b0); self.bbar = c(b0); self.Phi = c(Phi0); self.W = c(W0)
        self.H = c(np.linalg.inv(Phi0.T @ W0 @ Phi0))
        dg, db = self.dg, self.db
        self.Sg = np.zeros(dg); self.SGb = np.zeros(dg); self.SG = np.zeros((dg, db)); self.Om = np.zeros((dg, dg))
        self.A = np.zeros(1 + db + db * db); self.ntrunc = ctypes.c_long(0); self.nproj = 0

    def run(self, z, x, y):
        z, x, y = (np.ascontiguousarray(v, dtype=float) for v in (z, x, y)); m = len(y)
        self.nproj += _lib.sgmm_run(self.model, self.method, self.i, m, self.dg, self.db, _p(z), _p(x), _p(y), self.n0,
                                    *self.prm, _p(self.beta), _p(self.bbar), _p(self.Phi), _p(self.W), _p(self.H),
                                    _p(self.Sg), _p(self.SGb), _p(self.SG), _p(self.Om), _p(self.A), ctypes.byref(self.ntrunc))
        self.i += m

    def summary(self):
        n = self.i; db = self.db; b = self.bbar.copy()
        ghat = (self.Sg - self.SGb + self.SG @ b) / n; J = n * ghat @ self.W @ ghat
        WP = self.W @ self.Phi
        try:
            Jp = n * ghat @ (self.W - WP @ np.linalg.solve(self.Phi.T @ WP, WP.T)) @ ghat
        except np.linalg.LinAlgError:
            Jp = np.nan
        A0, A1, A2 = self.A[0], self.A[1:1 + db], self.A[1 + db:].reshape(db, db)
        VRS = (A2 - np.outer(A1, b) - np.outer(b, A1) + A0 * np.outer(b, b)) / n ** 2 / n
        if self.method == 0:
            V = self.H / n
        elif self.method == 1:
            PW = self.H @ self.Phi.T @ self.W; V = PW @ (self.Om / n) @ PW.T / n
        else:
            V = None
        return dict(bbar=b, V=V, VRS=VRS, J=J, Jp=Jp)


def pilot_one(M, seed, rep, n0, method):
    """Pilot of the Python implementation for a single replication (identical initialisation)."""
    import sgmm
    z, x, y = sgmm.chunk(M, seed, rep, -1, n0)
    W1 = np.linalg.inv(z.T @ z / n0)
    b, g, G = sgmm.gn_gmm(M, z, x, y, np.zeros(M.d), W1)
    r, _ = M.r(x, y, b); gz = z * r[:, None]
    Sg = gz.T @ gz / n0 + 1e-6 * np.eye(M.q); Sz = z.T @ z / n0 + 1e-6 * np.eye(M.q)
    return b, G, np.linalg.inv(Sz if method == "s2sls" else Sg)
