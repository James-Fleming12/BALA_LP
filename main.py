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

    def solve(self, lp: LinearProgram, y, v):
        """
        returns -min_{x∈Ω} L(x,y) using v=argmin_{x∈Ω}⟨c-A^Ty,x⟩+⟨y,b⟩ and the fact that g(⋅)=-L(v_{k+1},⋅)
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
    def __init__(self, rho):
        self.rho = rho

    def solve(self, lp: LinearProgram, z, w, y):
        """
        returns -min_{x∈Ω} L(x,y) using g_k(z) = -Lρ(w, y) - (1/2ρ)||z - y||^2
        """
        lagr = np.inner(lp.c, w) + np.inner(y, lp.b - lp.A @ w)
        penalty = (self.rho / 2) * np.linalg.norm(lp.b - lp.A @ w)**2

        aug_val = lagr + penalty

        right_term = (1 / (2 * self.rho)) * np.linalg.norm(z - y)**2
        return -aug_val - right_term

class PrimalOracle:
    """
    Solver for primal values given dual iterates of an LP of the form
    Minimize: c^Tx
    Subject To: Ax = b
    """
    def __init__(self):
        pass

    def solve(self, lp: LinearProgram, y):
        """
        returns argmin_{x∈Ω}L(x,y) for simple box constraints
        """
        return (lp.c - lp.A.T @ y < 0).astype(float)

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

        if np.linalg.norm(d) < 1e-10: # No direction to optimize along, return w_k
            return w_k

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
        self.lagr: PrimalOracle = PrimalOracle()
        self.dual_lagr: DualLagrange = DualLagrange()
        self.feas_lagr: FeasibleLagrangian = FeasibleLagrangian(rho)

        self.beta = beta # descent parameter
        self.rho = rho

        self.max_iters = 10

    def check(self, lp: LinearProgram, y, z, v, w):
        lhs = self.dual_lagr.solve(lp, y, v) - self.dual_lagr.solve(lp, z, v)
        rhs = self.dual_lagr.solve(lp, y, v) - self.feas_lagr.solve(lp, z, w, y)
        rhs = self.beta * rhs

        return lhs >= rhs
 
    def solve(self, lp: LinearProgram):
        x = lp.x
        y = np.zeros(lp.b.shape[0])

        w = x.copy()
        v = self.lagr.solve(lp, y)

        for _ in range(self.max_iters):
            w = self.aug_lagr.solve(lp, v, w, y) 
            z = y + self.rho * (lp.b - lp.A @ w)
            v = self.lagr.solve(lp, z)

            if self.check(lp, y, z, v, w):
                y = z
                x = w
            else:
                pass

        return x

def box_constrained_set():
    # problem of the form 
    # min c^Tx s.t. Ax=b, 0≤x≤1 (box constraints)
    c = np.array([1.0, 2.0])
    A = np.array([[1.0, 1.0]]) # with these initializations, optimal solution is x = (0.5, 0)
    b = np.array([0.5])

    x_init = np.array([1.0, 1.0])
    
    lp = LinearProgram(x_init, c, A, b)

    solver = BALA(rho=1.0, beta=0.3)

    res = solver.solve(lp)

    print(f"Primal Solution x: {res}")
    print(f"Constraint Violation: {np.linalg.norm(A @ res - b)}")
    print(f"Objective Value: {np.inner(c, res)}")

def main():
    box_constrained_set()

if __name__=="__main__":
    main()