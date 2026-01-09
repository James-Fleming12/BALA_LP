import numpy as np

class LinearProgram:
    """
    Variable Holder for LPs of the Form
    Minimize: c^Tx
    Subject To: Ax = b
    """
    def __init__(self, x: np.ndarray, c: np.ndarray, A: np.matrix, b: np.ndarray):
        self.c = c
        self.A = A
        self.b = b
        self.x = x

class AugmentedLagrangian:
    """
    Augmented Lagrangian solver for Linear Programs of form
    Minimize: c^Tx
    Subject To: A_eq x = b_eq

    Over a set Ω_k=conv(v_k,w_k)
    """
    def __init__(self, rho):
        self.rho = rho # penalty parameter

    def saturate(self, phi):
        if phi < 0:
            return 0
        if phi > 1:
            return 1
        return phi

    def solve(self, lp: LinearProgram, v_k, w_k, y_k):
        d = v_k - w_k
        denom = self.rho * np.linalg.norm(lp.A @ d, ord=2)**2

        temp_left = np.inner(lp.A.T @ (lp.b - lp.A @ w_k), d) # left numerator term
        temp_left = self.rho * temp_left
        temp_right = np.inner(lp.A.T @ y_k - lp.c, d) # right numerator term

        numer = temp_left + temp_right

        phi = numer/denom

        alpha = self.saturate(phi)

        return alpha * v_k + (1-alpha) * w_k

class BALA:
    """
    An implementation of BALA for Linear Programs that uses a straight line feasible set approximation
    """
    def __init__(self, rho):
        aug_lag = AugmentedLagrangian(rho)
        
    def solve(self, lp: LinearProgram):
        pass

def main():
    pass

if __name__=="__main__":
    main()