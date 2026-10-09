/* Single-replication SGMM (Algorithm 1) for the multiplicative exponential IV model
 *   g(X, b) = z * (y exp(-x'b) - 1),   G(X, b) = z psi',  psi = -y exp(-x'b) x,
 * with efficient weights, rank-two Woodbury update of H (Lemma 3.2(b)), averaging and the
 * running sums of the online J statistic.  Matrices are row-major.  Compiled with
 *   gcc -O3 -march=native -shared -fPIC -o libsgmm.so sgmm_c.c -lm
 */
#include <math.h>
#include <string.h>
#include <stdlib.h>

void sgmm_exp(long n, int dg, int db, const double *Z, const double *X, const double *Y,
              long n0, double gamma0, double a, double i0, double box,
              double *beta, double *bbar, double *Phi, double *W, double *H,
              double *Sg, double *SGb, double *SG, double *A0, double *A1, double *A2)
{
    double *g = malloc(dg * sizeof(double)), *Wg = malloc(dg * sizeof(double));
    double *Wz = malloc(dg * sizeof(double)), *u1 = malloc(db * sizeof(double));
    double *step = malloc(db * sizeof(double)), *u = malloc(db * sizeof(double));
    double *psi = malloc(db * sizeof(double)), *HU = malloc(2 * db * sizeof(double));
    double *Xs = malloc(2 * db * sizeof(double));
    for (long i = 1; i <= n; i++) {
        const double *z = Z + (i - 1) * dg, *x = X + (i - 1) * db; double y = Y[i - 1];
        double N = (double)(n0 + i - 1), gam = gamma0 * pow(i + i0, -a);
        /* ---- parameter step at the current iterate */
        double xb = 0; for (int k = 0; k < db; k++) xb += x[k] * beta[k];
        double r = y * exp(-xb) - 1.0;
        for (int j = 0; j < dg; j++) g[j] = z[j] * r;
        for (int j = 0; j < dg; j++) { double s = 0; const double *Wr = W + j * dg; for (int l = 0; l < dg; l++) s += Wr[l] * g[l]; Wg[j] = s; }
        for (int k = 0; k < db; k++) { double s = 0; for (int j = 0; j < dg; j++) s += Phi[j * db + k] * Wg[j]; u1[k] = s; }
        for (int k = 0; k < db; k++) { double s = 0; for (int l = 0; l < db; l++) s += H[k * db + l] * u1[l]; step[k] = s; }
        for (int k = 0; k < db; k++) { double v = beta[k] - gam * step[k]; beta[k] = v > box ? box : (v < -box ? -box : v); }
        /* ---- updates at the averaged iterate */
        double xbb = 0; for (int k = 0; k < db; k++) xbb += x[k] * bbar[k];
        double eb = y * exp(-xbb), rb = eb - 1.0;
        for (int k = 0; k < db; k++) psi[k] = -eb * x[k];
        double cz = 0;
        for (int j = 0; j < dg; j++) { double s = 0; const double *Wr = W + j * dg; for (int l = 0; l < dg; l++) s += Wr[l] * z[l]; Wz[j] = s; cz += z[j] * s; }
        for (int k = 0; k < db; k++) { double s = 0; for (int j = 0; j < dg; j++) s += Phi[j * db + k] * Wz[j]; u[k] = s; }
        double ell = N + rb * rb * cz;
        double C00 = -N * rb * rb / ell, C01 = N / ell, C11 = cz / ell;
        /* HU = H [u, psi]  (db x 2) */
        for (int k = 0; k < db; k++) { double s0 = 0, s1 = 0; for (int l = 0; l < db; l++) { s0 += H[k * db + l] * u[l]; s1 += H[k * db + l] * psi[l]; } HU[2 * k] = s0; HU[2 * k + 1] = s1; }
        double M00 = 0, M01 = 0, M11 = 0;   /* U'HU */
        for (int k = 0; k < db; k++) { M00 += u[k] * HU[2 * k]; M01 += u[k] * HU[2 * k + 1]; M11 += psi[k] * HU[2 * k + 1]; }
        /* T = I + C M / N */
        double T00 = 1 + (C00 * M00 + C01 * M01) / N, T01 = (C00 * M01 + C01 * M11) / N;
        double T10 = (C01 * M00 + C11 * M01) / N, T11 = 1 + (C01 * M01 + C11 * M11) / N;
        double det = T00 * T11 - T01 * T10;
        /* Xs = T^{-1} C (HU)'  (2 x db) */
        for (int k = 0; k < db; k++) {
            double a0 = C00 * HU[2 * k] + C01 * HU[2 * k + 1], a1 = C01 * HU[2 * k] + C11 * HU[2 * k + 1];
            Xs[k] = (T11 * a0 - T01 * a1) / det; Xs[db + k] = (-T10 * a0 + T00 * a1) / det;
        }
        double f = (N + 1) / N;
        for (int k = 0; k < db; k++) for (int l = 0; l < db; l++)
            H[k * db + l] = f * (H[k * db + l] - (HU[2 * k] * Xs[l] + HU[2 * k + 1] * Xs[db + l]) / N);
        for (int j = 0; j < dg; j++) for (int k = 0; k < db; k++) Phi[j * db + k] = (N * Phi[j * db + k] + z[j] * psi[k]) / (N + 1);
        double rb2 = rb * rb / ell;
        for (int j = 0; j < dg; j++) for (int l = 0; l < dg; l++) W[j * dg + l] = f * (W[j * dg + l] - rb2 * Wz[j] * Wz[l]);
        /* J sums (at bbar_{i-1}) */
        double pb = 0; for (int k = 0; k < db; k++) pb += psi[k] * bbar[k];
        for (int j = 0; j < dg; j++) { Sg[j] += z[j] * rb; SGb[j] += z[j] * pb; for (int k = 0; k < db; k++) SG[j * db + k] += z[j] * psi[k]; }
        for (int k = 0; k < db; k++) bbar[k] += (beta[k] - bbar[k]) / (double)i;
        /* random-scaling sums: sum_s s^2, sum_s s^2 bbar_s, sum_s s^2 bbar_s bbar_s' */
        double w2 = (double)i * (double)i; A0[0] += w2;
        for (int k = 0; k < db; k++) { A1[k] += w2 * bbar[k]; for (int l = 0; l < db; l++) A2[k * db + l] += w2 * bbar[k] * bbar[l]; }
    }
    free(g); free(Wg); free(Wz); free(u1); free(step); free(u); free(psi); free(HU); free(Xs);
}
