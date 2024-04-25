import torch
import numpy as np

class LogisticRegressionLoss(object):
    
    def __init__(self, lmd: float = 0.0) -> None:
       self.lmd = lmd
   
    def func(self, data, target, x):
        return np.mean(np.log(1 + np.exp( - data@x * target ))) + self.lmd/2 * np.linalg.norm(x)**2
    
    def grad(self, data, target, x) -> None:
        r = np.exp( - data@x * target )
        ry = -r/(1+r) * target
        return (data.T @ ry )/data.shape[0]  + self.lmd * x
    
    def hess(self, data, target, x):
        r = np.exp( - data@x * target )
        rr= r/(1+r)**2
        return (data.T@np.diagflat(rr)@data) / data.shape[0] + self.lmd*np.eye(data.shape[1])

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
