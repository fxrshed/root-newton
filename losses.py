import torch
import numpy as np

class BaseOracle(object):
    
    def func(self, x):
        raise NotImplementedError
    
    def grad(self, x):
        raise NotImplementedError
    
    def hess(self, x):
        raise NotImplementedError

class LogisticRegressionLoss(BaseOracle):
    
    def __init__(self, data, target, lmd: float = 0.0) -> None:
        self.data = data
        self.target = target
        self.lmd = lmd
   
    def func(self, x):
        return np.mean(np.log(1 + np.exp( - self.data@x * self.target ))) + self.lmd/2 * np.linalg.norm(x)**2
    
    def grad(self, x) -> None:
        r = np.exp( - self.data@x * self.target )
        ry = -r/(1+r) * self.target
        return (self.data.T @ ry )/self.data.shape[0]  + self.lmd * x
    
    def hess(self, x):
        r = np.exp( - self.data@x * self.target )
        rr= r/(1+r)**2
        return (self.data.T@np.diagflat(rr)@self.data) / self.data.shape[0] + self.lmd*np.eye(self.data.shape[1])

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
