A simple implementation of [BALA](https://arxiv.org/abs/2502.08835v1) for Equality-Constrained Linear Programs using the line segment feasible set approximation proposed in the paper. The only method that is problem constraint specific is the PrimalOracle calculation for v at each iteration, as the dual lagrange calculation uses the value of v to remove any necesitty for global information in its calculation.

Includes a stopping criterion based on the distance between the real Dual Lagrange value g(y) and the approximate Dual Lagrange Value g_k(y)

Currently the PrimalOracle code has support for the following constraint types:

- Box Constraints 0 ≤ x_i ≤ 1