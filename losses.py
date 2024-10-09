import torch
import numpy as np

class BaseOracle(object):
    
    def func(self):
        raise NotImplementedError
    
    def grad(self):
        raise NotImplementedError
    
    def hess(self):
        raise NotImplementedError

class LogisticRegressionLoss(BaseOracle):
    def __init__(self, data: np.ndarray, target: np.ndarray, lmd: float = 0.0) -> None:
        """Oracle for Logistic Regression Loss function. 
        Args:
            data (np.ndarray): data points 
            target (np.ndarray): targets 
            lmd (float, optional): regularization parameter. Defaults to 0.0.
        """
        self.data = data
        self.target = target
        self.lmd = lmd
        
        if self.lmd < 0.0:
            raise ValueError("Regularization parameter `lmd` cannot be negative.")
   
    def func(self, x: np.ndarray) -> np.floating:
        r"""Logistic Regression loss at point x. 
        
        .. math:: 
        `\min_{w \in \R^d }\Big\{ f(w) = \tfrac{1}{n}\sum_{i=1}^n \log(1 + \exp(-y_i \langle x_i, w\rangle)) + \frac{\mu}{2} \|w\|_2^2\Big\}.`

        Args:
            x (np.ndarray): data point to evaluate loss for. 

        Returns:
            np.floating: loss value at point x. 
        """
        return np.mean(np.log(1 + np.exp( - self.data@x * self.target ))) + self.lmd/2 * np.linalg.norm(x)**2
    
    def grad(self, x: np.ndarray) -> np.ndarray:
        """Logistic Regression gradient at point x.

        Args:
            x (np.ndarray): data point to evaluate gradient for.

        Returns:
            np.ndarray: gradient of loss function at point x.
        """
        r = np.exp( - self.data@x * self.target )
        ry = -r/(1+r) * self.target
        return (self.data.T @ ry )/self.data.shape[0]  + self.lmd * x
    
    def hess(self, x: np.ndarray) -> np.ndarray:
        """Logistic Regression hessian matrix at point x.

        Args:
            x (np.ndarray): data point to evaluate hessian for.

        Returns:
            np.ndarray: hessian matrix of loss function at point x.
        """
        r = np.exp( - self.data@x * self.target )
        rr= r/(1+r)**2
        return (self.data.T@np.diagflat(rr)@self.data) / self.data.shape[0] + self.lmd*np.eye(self.data.shape[1])


class PolytopeFeasibility(BaseOracle):
    
    def __init__(self, data, target, p):
        r"""Oracle for function:
        func(x) = sum_{i = 1}^m (<a_i, x> - b_i)_+^p,
        where (t)_+ := max(0, t).
        a_1, ..., a_m are rows of (m x n) matrix A.
        b is given (m x 1) vector.
        p >= 2 is a parameter.
        
        Args:
            data (_type_): data points 
            target (_type_): targets
            p (_type_): parameter
            
        Source: https://github.com/doikov/super-newton

        """
        self.A = data
        self.b = target
        self.p = p
        
        if self.p < 2:
            raise ValueError("Parameter `p` must be >= 2.")
        
        self.last_x = None

    def func(self, x):
        self._update_a(x)
        return np.sum(self.a ** self.p)

    def grad(self, x):
        self._update_a(x)
        return self.p * self.A.T.dot(self.a ** (self.p - 1))

    def hess(self, x):
        self._update_a(x)
        if self.p == 2:
            u = np.zeros_like(self.a)
            u[self.a > 0] = 1.0
            return 2 * self.A.T.dot(self.A * u.reshape(-1, 1))
        else:
            return self.p * (self.p - 1) * \
                self.A.T.dot(self.A * (self.a ** (self.p - 2)).reshape(-1, 1))

    def _update_a(self, x):
        if not np.array_equal(self.last_x, x):
            self.last_x = np.copy(x)
            self.a = self.A.dot(x) - self.b
            self.a = np.maximum(self.a, np.zeros_like(self.a))
   

# PyTorch Logistic Regression

def logreg(w, X, y, mu=0.0):
    return torch.mean(torch.log(1 + torch.exp(-y * (X @ w)))) + mu/2 * torch.norm(w)**2

def grad_logreg(w, X, y, mu=0.0):
    r = torch.exp(-y * (X @ w))
    return ( (r/(1 + r)) @ (X * -y[:, None]) ) / X.shape[0] + mu * w

def hess_logreg(w, X, y, mu=0.0):
    r = torch.exp(-y * (X @ w))
    return ( X.T @ (  (r/torch.square(1 + r)).reshape(-1, 1) * X ) ) / X.shape[0] + mu * torch.eye(X.shape[1])


# NumPy Logistic Regression

def lgstc(w, X, y, lmd=0.0):
  return np.mean(np.log(1 + np.exp( - X@w * y ))) + lmd/2 * np.linalg.norm(w)**2

def dlgstc(w, X, y, lmd=0.0):
  r = np.exp( - X@w * y )
  ry = -r/(1+r) * y
  return (X.T @ ry )/X.shape[0]  + lmd * w

def d2lgstc(w, X, y, lmd=0.0):
  r = np.exp( - X@w * y )
  rr= r/(1+r)**2
  return (X.T@np.diagflat(rr)@X) / X.shape[0] + lmd*np.eye(X.shape[1])
