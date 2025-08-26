import numpy as np

class SmallAnderson:
    def __init__(self, m=6, beta=1.0):
        self.m = m
        self.beta = beta
        self.mu_hist = []   # list of np.array vectors
        self.res_hist = []  # list of residuals (R(mu)-target)

    def push(self, mu_vec, res_vec):
        self.mu_hist.append(mu_vec.copy())
        self.res_hist.append(res_vec.copy())
        if len(self.mu_hist) > self.m:
            self.mu_hist.pop(0); self.res_hist.pop(0)

    def mix(self):
        k = len(self.res_hist)
        if k == 1:
            return self.mu_hist[-1] - self.beta * self.res_hist[-1]
        # form matrix of residual differences
        F = np.column_stack([r.flatten() for r in self.res_hist])
        # solve least squares for coefficients (regularize slightly)
        try:
            gamma, *_ = np.linalg.lstsq(F, F[:,-1], rcond=None)
            # build linear combo of previous mu's (simple variant)
            mu_comb = np.zeros_like(self.mu_hist[0])
            for i, mu in enumerate(self.mu_hist):
                mu_comb += mu * (1.0 / k)   # fallback equal weights
            # final damping
            return mu_comb - self.beta * self.res_hist[-1]
        except Exception:
            return self.mu_hist[-1] - self.beta * self.res_hist[-1]
