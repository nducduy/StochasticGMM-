/* General compiled engine for Algorithm 1 with IV-type moments g = z r(x, y, b), G = z psi'.
 *   model  0: linear       r = x'b - y,                psi = x
 *          1: exponential  r = y exp(-x'b) - 1,         psi = -y exp(-x'b) x
 *   method 0: SGMM (efficient weights, Gauss-Newton step, rank-two Woodbury update of H)
 *          1: online 2SLS (weights from z z', Gauss-Newton step)
 *          2: first-order (efficient weights, step Phi' W g without H)
 * The state (beta, bbar, Phi, W, H, running sums) is updated in place for observations
 * i = i_start+1, ..., i_start+m, so that the caller can stop at checkpoints.  Row-major storage.
 * Build: gcc -O3 -march=native -shared -fPIC -o libsgmm_engine.so sgmm_engine.c -lm
 */
#include <math.h>
#include <stdlib.h>

static inline void resid(int model, const double *x, double y, const double *b, int db, double *r, double *psi)
{
    double xb = 0; for (int k = 0; k < db; k++) xb += x[k] * b[k];
    if (model == 0) { *r = xb - y; if (psi) for (int k = 0; k < db; k++) psi[k] = x[k]; }
    else { double e = y * exp(-xb); *r = e - 1.0; if (psi) for (int k = 0; k < db; k++) psi[k] = -e * x[k]; }
}

long sgmm_run(int model, int method, long i_start, long m, int dg, int db,
              const double *Z, const double *X, const double *Y, long n0,
              double gamma0, double a, double i0, double box, double cP, double nu,
              double *beta, double *bbar, double *Phi, double *W, double *H,
              double *Sg, double *SGb, double *SG, double *Om, double *A, long *ntrunc)
{
    double *g = malloc(dg * sizeof(double)), *Wg = malloc(dg * sizeof(double)), *Wz = malloc(dg * sizeof(double));
    double *u1 = malloc(db * sizeof(double)), *step = malloc(db * sizeof(double)), *u = malloc(db * sizeof(double));
    double *psi = malloc(db * sizeof(double)), *HU = malloc(2 * db * sizeof(double)), *Xs = malloc(2 * db * sizeof(double));
    double *A1 = A + 1, *A2 = A + 1 + db;     /* A[0] = sum s^2, then sum s^2 bbar_s, then sum s^2 bbar bbar' */
    long nproj = 0;
    for (long t = 0; t < m; t++) {
        long i = i_start + t + 1;
        const double *z = Z + t * dg, *x = X + t * db; double y = Y[t];
        double N = (double)(n0 + i - 1), gam = gamma0 * pow((double)i + i0, -a), r, rb;
        /* ---- parameter step at the current iterate */
        resid(model, x, y, beta, db, &r, NULL);
        for (int j = 0; j < dg; j++) g[j] = z[j] * r;
        for (int j = 0; j < dg; j++) { double s = 0; const double *Wr = W + j * dg; for (int l = 0; l < dg; l++) s += Wr[l] * g[l]; Wg[j] = s; }
        for (int k = 0; k < db; k++) { double s = 0; for (int j = 0; j < dg; j++) s += Phi[j * db + k] * Wg[j]; u1[k] = s; }
        if (method == 2) { for (int k = 0; k < db; k++) step[k] = u1[k]; }
        else {
            for (int k = 0; k < db; k++) { double s = 0; for (int l = 0; l < db; l++) s += H[k * db + l] * u1[l]; step[k] = s; }
            double nh = 0, np_ = 0, nw = 0;     /* truncation: ||H|| ||Phi|| ||W|| (Frobenius) */
            for (int k = 0; k < db * db; k++) nh += H[k] * H[k];
            for (int k = 0; k < dg * db; k++) np_ += Phi[k] * Phi[k];
            for (int k = 0; k < dg * dg; k++) nw += W[k] * W[k];
            if (sqrt(nh * np_ * nw) > cP * pow((double)i, nu)) { for (int k = 0; k < db; k++) step[k] = 0; (*ntrunc)++; }
        }
        int proj = 0;
        for (int k = 0; k < db; k++) { double v = beta[k] - gam * step[k]; if (v > box) { v = box; proj = 1; } if (v < -box) { v = -box; proj = 1; } beta[k] = v; }
        nproj += proj;
        /* ---- updates at the averaged iterate */
        resid(model, x, y, bbar, db, &rb, psi);
        double rho = (method == 1) ? 1.0 : rb;        /* s = rho z */
        double cz = 0;
        for (int j = 0; j < dg; j++) { double s = 0; const double *Wr = W + j * dg; for (int l = 0; l < dg; l++) s += Wr[l] * z[l]; Wz[j] = s; cz += z[j] * s; }
        double ell = N + rho * rho * cz, f = (N + 1) / N;
        if (method != 2) {
            for (int k = 0; k < db; k++) { double s = 0; for (int j = 0; j < dg; j++) s += Phi[j * db + k] * Wz[j]; u[k] = s; }
            double C00 = -N * rho * rho / ell, C01 = N / ell, C11 = cz / ell;
            for (int k = 0; k < db; k++) { double s0 = 0, s1 = 0; for (int l = 0; l < db; l++) { s0 += H[k * db + l] * u[l]; s1 += H[k * db + l] * psi[l]; } HU[2 * k] = s0; HU[2 * k + 1] = s1; }
            double M00 = 0, M01 = 0, M11 = 0;
            for (int k = 0; k < db; k++) { M00 += u[k] * HU[2 * k]; M01 += u[k] * HU[2 * k + 1]; M11 += psi[k] * HU[2 * k + 1]; }
            double T00 = 1 + (C00 * M00 + C01 * M01) / N, T01 = (C00 * M01 + C01 * M11) / N;
            double T10 = (C01 * M00 + C11 * M01) / N, T11 = 1 + (C01 * M01 + C11 * M11) / N, det = T00 * T11 - T01 * T10;
            for (int k = 0; k < db; k++) {
                double a0 = C00 * HU[2 * k] + C01 * HU[2 * k + 1], a1 = C01 * HU[2 * k] + C11 * HU[2 * k + 1];
                Xs[k] = (T11 * a0 - T01 * a1) / det; Xs[db + k] = (-T10 * a0 + T00 * a1) / det;
            }
            for (int k = 0; k < db; k++) for (int l = 0; l < db; l++)
                H[k * db + l] = f * (H[k * db + l] - (HU[2 * k] * Xs[l] + HU[2 * k + 1] * Xs[db + l]) / N);
        }
        for (int j = 0; j < dg; j++) for (int k = 0; k < db; k++) Phi[j * db + k] = (N * Phi[j * db + k] + z[j] * psi[k]) / (N + 1);
        double q = rho * rho / ell;
        for (int j = 0; j < dg; j++) for (int l = 0; l < dg; l++) W[j * dg + l] = f * (W[j * dg + l] - q * Wz[j] * Wz[l]);
        /* running sums at bbar_{i-1}: online J and (for 2SLS) the moment covariance */
        double pb = 0; for (int k = 0; k < db; k++) pb += psi[k] * bbar[k];
        for (int j = 0; j < dg; j++) { Sg[j] += z[j] * rb; SGb[j] += z[j] * pb; for (int k = 0; k < db; k++) SG[j * db + k] += z[j] * psi[k]; }
        if (method == 1) for (int j = 0; j < dg; j++) for (int l = 0; l < dg; l++) Om[j * dg + l] += z[j] * z[l] * rb * rb;
        for (int k = 0; k < db; k++) bbar[k] += (beta[k] - bbar[k]) / (double)i;
        double w2 = (double)i * (double)i; A[0] += w2;
        for (int k = 0; k < db; k++) { A1[k] += w2 * bbar[k]; for (int l = 0; l < db; l++) A2[k * db + l] += w2 * bbar[k] * bbar[l]; }
    }
    free(g); free(Wg); free(Wz); free(u1); free(step); free(u); free(psi); free(HU); free(Xs);
    return nproj;
}
