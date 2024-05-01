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