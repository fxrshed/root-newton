import torch
torch.set_default_dtype(torch.float64)

import numpy as np
import pytest

from .. import losses
from .. import utils

@pytest.fixture
def synthetic_dataset():
    torch.manual_seed(0)
    
    X = torch.randn(1000, 1000)
    y = torch.randn(1000)
    y[y <= 0.0] = -1.0
    y[y > 0.0] = 1.0
    return X, y


def test_loss_func1(synthetic_dataset):
    X, y = synthetic_dataset
    w = torch.randn(1000, requires_grad=True)
    
    loss_t = losses.logreg(w, X, y)

    X = X.numpy()
    y = y.numpy()
    w = w.detach().numpy()
    
    loss_function = losses.LogisticRegressionLoss(X, y)
    loss = loss_function.func(w)
    
    assert loss - loss_t <= 1e-12
    
def test_loss_func2(synthetic_dataset):
    X, y = synthetic_dataset
    w = torch.randn(1000)
    
    X = X.numpy()
    y = y.numpy()
    w = w.numpy()
    
    loss_t = losses.lgstc(w, X, y)
    
    loss_function = losses.LogisticRegressionLoss(X, y)
    loss = loss_function.func(w)
    
    assert loss - loss_t <= 1e-12

def test_loss_grad1(synthetic_dataset):
    X, y = synthetic_dataset
    w = torch.randn(1000, requires_grad=True)
    
    loss_t = losses.logreg(w, X, y)
    grad_t = torch.autograd.grad(loss_t, w)[0]

    X = X.numpy()
    y = y.numpy()
    w = w.detach().numpy()
    
    loss_function = losses.LogisticRegressionLoss(X, y)
    grad = loss_function.grad(w)

    assert grad_t.shape == grad.shape
    assert np.linalg.norm(grad_t.numpy() - grad) <= 1e-12
    
def test_loss_grad2(synthetic_dataset):
    X, y = synthetic_dataset
    w = torch.randn(1000)
    
    X = X.numpy()
    y = y.numpy()
    w = w.numpy()
    
    grad_t = losses.dlgstc(w, X, y)

    loss_function = losses.LogisticRegressionLoss(X, y)
    grad = loss_function.grad(w)

    assert grad_t.shape == grad.shape
    assert np.linalg.norm(grad_t - grad) <= 1e-12
    
def test_loss_hess1(synthetic_dataset):
    X, y = synthetic_dataset
    w = torch.randn(1000, requires_grad=True)
    
    loss_t = losses.logreg(w, X, y)
    closure = lambda w: losses.logreg(w, X, y)
    hess_t = torch.autograd.functional.hessian(closure, w)

    X = X.numpy()
    y = y.numpy()
    w = w.detach().numpy()
    
    loss_function = losses.LogisticRegressionLoss(X, y)
    hess = loss_function.hess(w)
    
    assert hess_t.shape == hess.shape
    assert np.linalg.norm(hess_t.numpy() - hess) <= 1e-12
    
def test_loss_hess2(synthetic_dataset):
    X, y = synthetic_dataset
    w = torch.randn(1000)
    
    X = X.numpy()
    y = y.numpy()
    w = w.numpy()
    
    hess_t = losses.d2lgstc(w, X, y)
    
    loss_function = losses.LogisticRegressionLoss(X, y)
    hess = loss_function.hess(w)
    
    assert hess_t.shape == hess.shape
    assert np.linalg.norm(hess_t - hess) <= 1e-12
    
    
    
    
    
    
    
    
# def pol_f(A, x, b, p):
#     a = A.dot(x) - b
#     a = np.maximum(a, np.zeros_like(a))
#     return np.sum(a ** p)

# def pol_f_t(A, x, b, p):
#     a = A @ x - b
#     a = torch.maximum(a, torch.zeros_like(a))
#     return torch.sum(a ** p)

# np.random.seed(0)

# n = 200 # Number of linear inequalities
# d = 100 # Dimension
# p = 3 # Smoothing parameter

# A = np.random.rand(n, d) * 2 - 1
# x_star = np.random.rand(d) * 2 - 1
# x_0 = np.ones(d)
# b = A.dot(x_star)
# oracle = PolytopeFeasibility(A, b, p)

# loss = oracle.func(x_0)
# grad = oracle.grad(x_0)
# hess = oracle.hess(x_0)


# A = torch.tensor(A)
# x_star = torch.tensor(x_star)
# x_0 = torch.tensor(x_0).requires_grad_()
# b = torch.tensor(b)

# loss_t = pol_f_t(A, x_0, b, p)
# grad_t = torch.autograd.grad(loss_t, x_0)[0]
# closure = lambda x: pol_f_t(A, x, b, p)
# hess_t = torch.autograd.functional.hessian(closure, x_0)

# loss_t - loss, torch.norm(grad_t - torch.tensor(grad)), torch.norm(hess_t - torch.tensor(hess))