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

class DualLagrange:
    def __init__(self):
        pass

    def solve(self, lp: LinearProgram, v, y):
        """
        returns -min_{x∈Ω} L(x,y) using v=argmin_{x∈Ω}⟨c-A^Ty,x⟩+⟨y,b⟩
        """
        left_term = np.inner(lp.c, v)
        right_term = np.inner(y, lp.b - lp.A @ v)
        return -left_term - right_term

class FeasibleLagrangian:
    """
    Standard Lagrangian solver for Linear Programs of form
    Minimize: c^Tx
    Subject To: Ax = b

    Over a set Ω_k=conv(v_k,w_k)
    """
    def __init__(self):
        pass

    def solve(self, lp: LinearProgram, y, w):
        """
        returns -min_{x∈Ω} L(x,y)
        """
        left_term = np.inner(lp.c, w)
        right_term = np.inner(y, lp.b - lp.A @ w)
        return -left_term - right_term

class Lagrangian:
    """
    Standard Lagrangian solver for Linear Programs of form
    Minimize: c^Tx
    Subject To: Ax = b
    """
    def __init__(self):
        pass

    def solve(self, lp: LinearProgram, y):
        """
        returns argmin_{x∈Ω}L(x,y)
        """
        pass

class AugmentedLagrangian:
    """
    Augmented Lagrangian solver for Linear Programs of form
    Minimize: c^Tx
    Subject To: Ax = b

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
    def __init__(self, rho, beta):
        self.aug_lagr: AugmentedLagrangian = AugmentedLagrangian(rho)
        self.lagr: Lagrangian = Lagrangian()
        self.dual_lagr: DualLagrange = DualLagrange()
        self.feas_lagr: FeasibleLagrangian = FeasibleLagrangian()

        self.beta = beta # descent parameter
        self.rho = rho

        self.max_iters = 1000

    def check(self, lp: LinearProgram, y, z, v, w):
        lhs = self.dual_lagr.solve(lp, y, v) - self.dual_lagr.solve(lp, z, v)
        rhs = self.dual_lagr.solve(lp, y, v) - self.feas_lagr.solve(lp, z, w)
        rhs = self.beta * rhs

        return lhs >= rhs
 
    def solve(self, lp: LinearProgram):
        x = lp.x
        y = np.zeros(lp.b.shape[0])

        w = x.copy()
        v = self.lagr.solve(lp, y)

        for i in range(self.max_iters):
            w = self.aug_lagr.solve(lp, w, v, y) # primal candidate
            z = y + self.rho * (lp.b - lp.A @ w) # dual candidate

            if self.check(lp, y, z, v, w): # serious step
                x = w
                y = z
            else:
                pass # null step (nothing is updated)
            
            v = self.lagr.solve(lp, z)

def main():
    pass

if __name__=="__main__":
    main()