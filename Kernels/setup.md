Implement two regression methods, (1) kernelized ridge regression and (2) Support Vector Regression (SVR), and two kernels:

- Polynomial kernel $\kappa(x,x') = (1 + xx')^M$
- RBF kernel $\kappa(x,x') = \mathrm{exp}(-\frac{||x-x'||^2}{2\sigma^2})$

Implement SVR by solving the optimization problem in Eq. (10) from (Smola and Scholkopf, 2004) with cvxopt.solvers.qp. Inputs to qp should be represented so that the solution x contains $\alpha_i$ and $\alpha_i^* $ in the following order: $[\alpha_1, \alpha_1^*, \alpha_2, \alpha_2^*, \alpha_3, \alpha_3^*, \dots]$.
Set $C$ as $1/\lambda$.

Hint. Computation of $b$ needs to support kernels, so use $\sum_{j=1}^{l}(\alpha_j - \alpha_j^*)\langle x_j, x_i \rangle$ instead of $\langle w, x_i \rangle$. Also, results from qp are approximate; take care to handle this. Then, Eq. (16) from (Smola and Scholkopf, 2004) is probably wrong. In the $\mathrm{min}$ part there, I think, should be $+\epsilon$ instead of $-\epsilon$.

Hint. For the large majority of cases, the inequalities of Eq. (16) should collapse (up to some small error) in a correct solution. They are only problematic when there is some optimization artifact in qr (like having a very small lambda, which causes large C). For these rare cases perhaps take the mean of the limits.

Apply both regression methods and both kernels to the 1-dimensional sine data set. For each method/kernel find kernel and regularization parameters that work well. For SVR, also take care to produce a good fitting sparse solution. This part aims to showcase what kernels can do and introduce the meaning of parameters. No need to do any formal parameter selection (such as with cross-validation) here. Plot the input data, the fit, and mark support vectors on the plot.

Apply both regression methods and both kernels to the housing2r data set. Use the first 80% of data as a training set and the remaining 20% as a validation set. For each method/kernel, plot RMSE on the testing set versus a kernel parameter value (for polynomial kernel, M $\in$ $[1, 10]$, for RBF choose interesting values of $\sigma$ yourself). Take care to set $\epsilon$ properly. Plot two curves for each kernel/method, one with regularization parameter $\lambda=1$, and the other with $\lambda$ set with internal cross validation (for each kernel parameter value separately). For SVR, also display the number of support vectors for each score and try to keep it to a minimum while still getting a good fit.

Compare results between kernelized ridge regression and SVR and comment on the differences and similarities. Which learning algorithm would you prefer and why?

Submit your code in a single file named hw_kernels.py and a max two page report (.pdf) with plots, chosen parameters and comments. Your code has to be Python 3.8 compatible and must conform to the unit tests from test_kernels.py (also see comments therein for implementation details). Your solution to regression methods and kernels can only use the python standard library, numpy, cvxopt libraries. For data input, visualizations, and model evaluation (cross validation, scoring) you may use any additional libraries.
