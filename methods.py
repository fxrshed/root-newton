import numpy as np
import scipy
from losses import BaseOracle


class BaseOptimizer(object):
    def __init__(self, params: np.ndarray):
        """Base class for Optimizers.

        Args:
            params (np.ndarray): model parameters.
        """
        self.params = params
    
    def step(self, oracle: BaseOracle):
        raise NotImplementedError
    
    
def newton_search(hess: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Returns Newton search direction calculation procedure to be used across all optimization methods.

    Args:
        hess (np.ndarray): Hessian matrix of the loss function
        grad (np.ndarray): gradient vector of the loss function 

    Returns:
        np.ndarray: Newton search direction :math:`[\nabla^2 F(x)]^{-1} \nabla F(x)`.
    """
    hess_inv = np.linalg.pinv(hess, rcond=1e-10)
    return hess_inv @ grad

class GradRegNewton(BaseOptimizer):
    def __init__(self, params: np.ndarray, q: float = 2.0, L_est: float = 1.0, verbose: bool = False):
        r"""Implements Gradient Regularization of Newton method proposed in ``Super-Universal Regularized Newton Method``

        Args:
            params (np.ndarray): model parameters
            q (float, optional): :math:`q = \nu + p \in [2, 4]`. Refer to Hölder continuity in Section 3 of [1]. Defaults to 2.0.
            L_est (float, optional): :math:`\M_q > 0` in [1]. Defaults to 1.0. 

        References:
        [1] ``Super-Universal Regularized Newton Method``: https://arxiv.org/pdf/2208.05888
        """
        super().__init__(params)
        self.q = q
        self.L_est = L_est
        self.verbose = verbose
        
        if self.q < 2 or self.q > 4:
            raise ValueError("Parameter `q` must me in range [2.0, 4.0].")
        
    def step(self, oracle: BaseOracle) -> np.ndarray:
        """Performs single optimization step. 

        Args:
            oracle (BaseOracle): oracle instance with first and second order information.

        Returns:
            np.ndarray: model parameters after performing optimization step.
        """
    
        grad = oracle.grad(self.params)
        hess = oracle.hess(self.params)
        B = np.eye(self.params.shape[0])
        try:
            n = newton_search(hess, grad)
        except (np.linalg.LinAlgError, ValueError) as e:
            if self.verbose:
                print(f'Warning: linalg_error: {e}', flush=True)
            return self.params
        
        g = np.sqrt(grad.dot(n))
        lambda_k = (6 * self.L_est * g**(self.q - 2))**(1 / (self.q - 1))
        try:
            # Compute the regularized Newton step
            delta_w = scipy.linalg.cho_solve(scipy.linalg.cho_factor(
                            hess + lambda_k * B, lower=False), -grad)
        except (np.linalg.LinAlgError, ValueError) as e:
            if self.verbose:
                print(f'Warning: linalg_error: {e}', flush=True)
            return self.params
            
        self.params += delta_w
        
        return self.params
    
    
class AICN(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, L_est: float = 1.0, verbose: bool = False):
        r"""Implements  Affine-Invariant Cubic Newton (AICN) proposed in ``A Damped Newton Method Achieves Global O(1/k^2) and Local Quadratic Convergence Rate``.

        Args:
            params (np.ndarray): model parameters
            L_est (float, optional): positive constant s.t. :math:`L_{est} \geq L_{semi}`. Defaults to 1.0.
            
        References:
        [1] ``A Damped Newton Method Achieves Global O(1/k^2) and Local Quadratic Convergence Rate``: https://arxiv.org/pdf/2211.00140.
        """
        super().__init__(params)
        self.L_est = L_est
        self.verbose = verbose
        
    def step(self, oracle: BaseOracle) -> np.ndarray:
        """Performs single optimization step.

        Args:
            oracle (BaseOracle): oracle instance with first and second order information.

        Returns:
            np.ndarray:  model parameters after performing optimization step.
        """
        grad = oracle.grad(self.params)
        hess = oracle.hess(self.params)
        try:
            n = newton_search(hess, grad)
        except (np.linalg.LinAlgError, ValueError) as e:
            if self.verbose:
                print(f'Warning: linalg_error: {e}', flush=True)
            return self.params
        
        g = grad.dot(n) ** 0.5
        lr = (np.sqrt(1 + 2 * self.L_est * g) - 1) / (self.L_est * g)
        self.params -= lr * n

        return self.params
    
class RootNewton(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, q: float, L_est: float = 1.0, verbose: bool =False):
        r"""Implements Root Newton method proposed in ``Better global convergence guarantees for stepsized
Newton method``

        Args:
            params (np.ndarray): model parameters.
            q (float): :math:`q = \nu + p \in [2, 4]`. Refer to Hölder continuity in [1]. Defaults to 2.0.
            L_est (float, optional): :math:`L_est := M_q := L_{p, \nu}`. Defaults to 1.0.
        """
        super().__init__(params)
        self.q = q
        self.L_est = L_est
        self.verbose = verbose
        
        if self.q < 2.0 or self.q > 4.0:
            raise ValueError("Parameter `q` must be in range [2.0, 4.0].")
        
        if self.L_est <= 0.0:
            raise ValueError("`L` estimation must be a positive value.")
        
    def step(self, oracle: BaseOracle) -> np.ndarray:
        """Performs single optimization step.

        Args:
            oracle (BaseOracle): oracle instance with first and second order information.

        Returns:
            np.ndarray:  model parameters after performing optimization step.
        """
        
        grad = oracle.grad(self.params)
        hess = oracle.hess(self.params)
        # n = newton_search(hess, grad)
        try:
            n = newton_search(hess, grad)
        except (np.linalg.LinAlgError, ValueError) as e:
            if self.verbose:
                print(f'Warning: linalg_error: {e}', flush=True)
            return self.params

        g = np.sqrt(grad.dot(n))
        theta = (9 * self.L_est)**(1 / (self.q - 1)) * g**((self.q - 2) / (self.q - 1))
        lr = 1 / (1 + theta)
        self.params -= lr * n

        return self.params
    
    
class SimpliReg(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, beta: float = 0.1, sigma: float = 0.1):
        super().__init__(params)
        self.beta = beta
        self.sigma = sigma
        
    def step(self, oracle: BaseOracle):
        
        grad = oracle.grad(self.params)
        hess = oracle.hess(self.params)
        try:
            n = newton_search(hess, grad)
        except (np.linalg.LinAlgError, ValueError) as e:
            if self.verbose:
                print(f'Warning: linalg_error: {e}', flush=True)
            return self.params
        
        g = np.sqrt(grad.dot(n))
        theta = (self.sigma + 1) * g**self.beta
        lr = (1 / theta)**(1 / 1 + self.beta)
        
        self.params -= lr * n

        return self.params
    
    
class GradientMethod(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, L_est: float = 1.0, 
                 L_min: float = 1e-5, verbose: bool = False):
        r"""Implements Gradient Method with backtracking line search routine from [1]. 

        Args:
            params (np.ndarray): model parameters
            L_est (float, optional): Initial estimation of :math:`L` constant. Defaults to 1.0.
            L_min (float, optional): Minimum value of :math:`L` constant. Defaults to 1e-5.
            verbose (bool, optional): Set to `True` to enable warnings. Defaults to False.
            
        References:
        [1] Yurii Nesterov. Lectures on convex optimization, volume 137. Springer, 2018.
        
        Code source: https://github.com/doikov/super-newton
        """
        super().__init__(params)
        self.L_k = L_est
        self.L_min = L_min
        self.verbose = verbose
        
    def step(self, oracle: BaseOracle) -> np.ndarray:
        """Performs single optimization step.

        Args:
            oracle (BaseOracle): oracle instance with zeroth and first order information.

        Returns:
            np.ndarray:  model parameters after performing optimization step.
        """
        
        func = oracle.func(self.params)
        grad = oracle.grad(self.params)

        l2_norm_sqr = lambda x: x.dot(x)
        
        line_search_max_iter = 30
        for i in range(line_search_max_iter + 1):
            if i == line_search_max_iter:
                if self.verbose:
                    print('Warning: line_search_max_iter_reached', flush=True)
                break

            T = self.params - grad / self.L_k
            func_T = oracle.func(T)

            if func_T <= func + grad.dot(T - self.params) + \
                            0.5 * self.L_k * l2_norm_sqr(T - self.params):
                self.L_k *= 0.5
                self.L_k = max(self.L_k, self.L_min)
                if self.verbose:
                    print(f"Line search took {i} steps: L_k = {self.L_k}")
                break

            self.L_k *= 2

        self.params -= grad / self.L_k

        return self.params
    
    
    
class SuperNewton(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, H_0: float = 1.0, 
                 alpha: float = 1.0, H_min: float = 1e-5,
                 verbose: bool = False):
        r"""Implements Super-Universal Newton Method from ``Super-Universal Regularized
Newton Method``[1]. 

        Args:
            params (np.ndarray): model parameters.
            H_0 (float, optional): Initial value of :math:`H_k`. Defaults to 1.0.
            alpha (float, optional): :math:`\alpha \in [2/3, 1]`. Defaults to 1.0.
            H_min (float, optional): Minimum value of :math:`H_k`. Defaults to 1e-5.
            verbose (bool, optional): Set to `True` to enable warnings. Defaults to False.
            
        References: 
        [1] ``Super-Universal Regularized
Newton Method``: https://arxiv.org/pdf/2208.05888
        
        Code source: https://github.com/doikov/super-newton
        """
        super().__init__(params)
        self.H_k = H_0
        self.H_min = H_min
        self.alpha = alpha
        self.B = np.eye(params.shape[0])
        self.verbose = verbose
        
        self.adaptive_search_max_iter = 40
        
    def step(self, oracle: BaseOracle) -> np.ndarray:
        """Performs single optimization step.

        Args:
            oracle (BaseOracle): oracle instance with first and second order information.

        Returns:
            np.ndarray:  model parameters after performing optimization step.
        """
        
        grad = oracle.grad(self.params)
        hess = oracle.hess(self.params)
        
        grad_norm = grad.dot(grad) ** 0.5
        
        for i in range(self.adaptive_search_max_iter + 1):
            if i == self.adaptive_search_max_iter:
                if self.verbose:
                    print(('Warning: adaptive_iterations_exceeded'), flush=True)
                break

            lambda_k = (4 ** i) * self.H_k * (grad_norm ** self.alpha)
            try:
                # Compute the regularized Newton step
                # n = scipy.linalg.cho_solve(scipy.linalg.cho_factor(
                #                 hess + lambda_k * self.B, lower=False), grad)
                n = newton_search(hess + lambda_k * self.B, grad)
            except (np.linalg.LinAlgError, ValueError) as e:
                if self.verbose:
                    print('Warning: linalg_error', flush=True)

            # loss_new = oracle.func(self.params - n)
            grad_new = oracle.grad(self.params - n)
            grad_norm_new_sqrd = grad_new.dot(grad_new) # squared norm of gradient at (w + delta_w) 

            # Check condition for H_k
            if grad_new.dot(n) >= grad_norm_new_sqrd / (4 * lambda_k):
                # self.H_k *= 0.25
                self.H_k = (4 ** i) * (self.H_k) * 0.25
                # self.H_k = max(self.H_k, self.H_min)
                break
            
        # Update the point
        self.params -= n
        
        return self.params


class UniversalNewton(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, beta: float = 1.0, sigma_0: float = 1.0, 
                 c: float = 1.0, verbose: bool = False):
        r"""Implements Universal Newton Step-size Schedule from ``Better global convergence guarantees for stepsized
Newton method``[1]. 

        Args:
            params (np.ndarray): model parameters.
            beta (float, optional): :math:`\beta \in [2/3, 1]`. Defaults to 1.0.
            sigma_0 (float, optional): Initial value of :math:`\sigma > 0`. Defaults to 1.0.
            c (float, optional): :math:`c > 0`. Defaults to 1.0.
            verbose (bool, optional): Set to `True` to enable warnings. Defaults to False.
        """
        
        super().__init__(params)
        self.beta = beta
        self.sigma_k = sigma_0
        self.c = c
        self.verbose = verbose
        
        if self.beta < 2/3 or self.beta > 1.0:
            raise ValueError("Parameter `beta` must be in range [2/3, 1].")
        
        if self.sigma_k <= 0.0:
            raise ValueError("Parameter `sigma` must be a postive value.")
        
        if self.c <= 0.0:
            raise ValueError("Parameter `c` must be a postive value.")
        
        self.adaptive_search_max_iter = 100
        
    def step(self, oracle: BaseOracle) -> np.ndarray:
        """Performs single optimization step.

        Args:
            oracle (BaseOracle): oracle instance with first and second order information.

        Returns:
            np.ndarray:  model parameters after performing optimization step.
        """
        
        grad = oracle.grad(self.params)
        hess = oracle.hess(self.params)
        # n = newton_search(hess, grad)
        try:
            n = newton_search(hess, grad)
        except (np.linalg.LinAlgError, ValueError) as e:
            if self.verbose:
                print('Warning: linalg_error', flush=True)
            return self.params
                
        g = grad.dot(n) ** 0.5
        
        for j in range(self.adaptive_search_max_iter + 1):
            if j == self.adaptive_search_max_iter:
                if self.verbose:
                    print(('Warning: adaptive_iterations_exceeded'), flush=True)
                break
            
            theta = self.c**j * self.sigma_k * g**self.beta
            alpha = 1 / (1 + theta)
            w = self.params - alpha * n

            # Check condition for H_k
            grad_j = oracle.grad(w)
            
            try:
                n_j = newton_search(hess, grad_j)
            except (np.linalg.LinAlgError, ValueError) as e:
                if self.verbose:
                    print('Warning: linalg_error', flush=True)
                    
            g_j_sq = grad_j.dot(n_j)
            if grad_j.dot(n) * (2 * alpha * theta) >= g_j_sq :
                self.sigma_k *= self.c**(j - 1)
                if self.verbose:
                    print(f"Line search took {j} steps: lr={alpha}")
                break
        
        # Update the parameters
        self.params -= alpha * n
            
        return self.params
        
        
class ArmijoNewton(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, gamma: float = 1.0, tau: float = 0.5, verbose: bool = False):
        super().__init__(params)
        self.gamma = gamma
        self.tau = tau
        self.lr = gamma
        self.verbose = verbose
        
        self.adaptive_search_max_iter = 100
        
    def step(self, oracle: BaseOracle) -> np.ndarray:
        """Performs a single optimization step.

        Args:
            oracle (BaseOracle): oracle instance with first and second order information.

        Returns:
            np.ndarray:  model parameters after performing optimization step.
        """
        
        loss = oracle.func(self.params)
        grad = oracle.grad(self.params)
        hess = oracle.hess(self.params)
        n = newton_search(hess, grad)
        d = -1.0 * n
        
        for j in range(1, self.adaptive_search_max_iter + 1):
            if j == self.adaptive_search_max_iter:
                if self.verbose:
                    print(('Warning: adaptive_iterations_exceeded'), flush=True)
                break
            
            self.lr = self.gamma**j
            new_params = self.params + self.lr * d
            new_loss = oracle.func(new_params)
            
            if new_loss <= loss + self.lr * self.tau * d.dot(grad):
                if self.verbose:
                    print(f"Armijo backtracking took {j} steps: lr={self.lr}")
                break

        # Update the parameters
        self.params += self.lr * d

        return self.params
    
    
class DampedNewton(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, lr: float = 1.0):
        super().__init__(params)
        self.lr = lr 
        
    def step(self, oracle: BaseOracle) -> np.ndarray:
        """Performs single optimization step.

        Args:
            oracle (BaseOracle): oracle instance with first and second order information.

        Returns:
            np.ndarray:  model parameters after performing optimization step.
        """
        
        grad = oracle.grad(self.params)
        hess = oracle.hess(self.params)
        n = newton_search(hess, grad)
        
        self.params -= self.lr * n
            
        return self.params
    
class CGNewton(BaseOptimizer):
    
    def __init__(self, params: np.ndarray, lr: float = 1.0):
        super().__init__(params)
        self.lr = lr 
        
    def step(self, oracle: BaseOracle) -> np.ndarray:
        """Performs single optimization step.

        Args:
            oracle (BaseOracle): oracle instance with first and second order information.

        Returns:
            np.ndarray:  model parameters after performing optimization step.
        """
        
        grad = oracle.grad(self.params)
        hess = oracle.hess(self.params)
        n, exit_code = scipy.sparse.linalg.cg(hess, grad)
        
        self.params -= self.lr * n
            
        return self.params
    
class GreedyNewton(BaseOptimizer):
    
    def __init__(self, 
                 params: np.ndarray, 
                 lr_range: tuple[float, float] = (1e-10, 1.0, 400), 
                 verbose: bool = False):
        super().__init__(params)
        self.verbose = verbose
        
        self.lrs = np.geomspace(lr_range[0], lr_range[1], lr_range[2])
        
    def step(self, oracle: BaseOracle) -> np.ndarray:
        """Performs single optimization step.

        Args:
            oracle (BaseOracle): oracle instance with first and second order information.

        Returns:
            np.ndarray:  model parameters after performing optimization step.
        """
        
        loss = oracle.func(self.params)
        grad = oracle.grad(self.params)
        hess = oracle.hess(self.params)
        
        n = newton_search(hess, grad)
        d = -1.0 * n
        
        best_lr = 0.0
        min_loss = oracle.func(self.params + best_lr * d)
        for lr in self.lrs:
            new_loss = oracle.func(self.params + lr * d)
            if new_loss < min_loss:
                min_loss = new_loss
                best_lr = lr
        
        self.lr = best_lr
        
        if self.verbose:
            print(f"{best_lr=}, {min_loss=}")
            
        # Update the parameters
        self.params += self.lr * d

        return self.params
    

class GRLS(BaseOptimizer):
    
    def __init__(self, 
                 params: np.ndarray, 
                 lr_range: tuple[float, float] = (1e-10, 1.0, 400), 
                 verbose: bool = False):
        super().__init__(params)
        self.verbose = verbose
        
        self.lrs = np.geomspace(lr_range[0], lr_range[1], lr_range[2])
        
    def step(self, oracle: BaseOracle) -> np.ndarray:
        """Performs single optimization step.

        Args:
            oracle (BaseOracle): oracle instance with first and second order information.

        Returns:
            np.ndarray:  model parameters after performing optimization step.
        """
        
        loss = oracle.func(self.params)
        grad = oracle.grad(self.params)
        hess = oracle.hess(self.params)
        
        # Not using `newton_search` method since we use `hess_inv` during linesearch
        hess_inv = np.linalg.pinv(hess)
        
        # Search direction
        d = -1.0 * (hess_inv @ grad)
        
        best_lr = 0.0
        new_params = self.params + best_lr * d
        new_grad = oracle.grad(new_params)
        ny_norm_squared = new_grad @ (hess_inv @ new_grad)
        min_expr = (oracle.func(new_params) - oracle.func(self.params)) / ny_norm_squared
        
        for lr in self.lrs:

            new_params = self.params + lr * d
            new_grad = oracle.grad(new_params)
            ny_norm_squared = new_grad @ (hess_inv @ new_grad)

            new_expr = (oracle.func(new_params) - oracle.func(self.params)) / ny_norm_squared
            
            if new_expr < min_expr:
                min_expr = new_expr.copy()
                best_lr = lr
        
        self.lr = best_lr
        
        if self.verbose:
            print(f"{best_lr=}, {min_expr=}")

        # Update the parameters
        self.params += self.lr * d

        return self.params