import torch
torch.set_default_dtype(torch.float64)

import numpy as np
import pytest

import losses
import utils

@pytest.fixture
def synthetic_dataset():
    torch.manual_seed(0)
    
    X = torch.randn(1000, 1000)
    y = torch.randn(1000)
    y[y <= 0.0] = -1.0
    y[y > 0.0] = 1.0
    return X, y


def test_loss_func(synthetic_dataset):
    X, y = synthetic_dataset
    w = torch.randn(1000, requires_grad=True)
    
    loss_t = losses.logreg(w, X, y)

    X = X.numpy()
    y = y.numpy()
    w = w.detach().numpy()
    
    loss_function = losses.LogisticRegressionLoss()
    loss = loss_function.func(X, y, w)
    
    assert loss - loss_t <= 1e-12

def test_loss_grad(synthetic_dataset):
    X, y = synthetic_dataset
    w = torch.randn(1000, requires_grad=True)
    
    loss_t = losses.logreg(w, X, y)
    grad_t = torch.autograd.grad(loss_t, w)[0]

    X = X.numpy()
    y = y.numpy()
    w = w.detach().numpy()
    
    loss_function = losses.LogisticRegressionLoss()
    grad = loss_function.grad(X, y, w)

    assert np.linalg.norm(grad_t.numpy() - grad) <= 1e-12
    
def test_loss_hess(synthetic_dataset):
    X, y = synthetic_dataset
    w = torch.randn(1000, requires_grad=True)
    
    loss_t = losses.logreg(w, X, y)
    closure = lambda w: losses.logreg(w, X, y)
    hess_t = torch.autograd.functional.hessian(closure, w)

    X = X.numpy()
    y = y.numpy()
    w = w.detach().numpy()
    
    loss_function = losses.LogisticRegressionLoss()
    hess = loss_function.hess(X, y, w)
    
    assert np.linalg.norm(hess_t.numpy() - hess) <= 1e-12
    
def test_dataset_map_classes_1():
    targets = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    new_classes = [-1.0, 1.0]
    assert np.array_equal(np.unique(utils.map_classes_to(targets, new_classes)), new_classes)
    
def test_dataset_map_classes_2():
    targets = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    new_classes = [1.0, 2.0]
    assert np.array_equal(np.unique(utils.map_classes_to(targets, new_classes)), new_classes)

def test_dataset_map_classes_3():
    targets = np.asarray([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0])
    new_classes = [1.0, 4.0]
    assert np.array_equal(np.unique(utils.map_classes_to(targets, new_classes)), new_classes)
    
def test_dataset_map_classes_4():
    targets = np.asarray([0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0])
    new_classes = [-1.0, 1.0, 3.0]
    assert np.array_equal(np.unique(utils.map_classes_to(targets, new_classes)), new_classes)
    
def test_dataset_map_classes_5():
    targets = np.asarray([0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0])
    new_classes = [-1.0, 1.0]
    with pytest.raises(AssertionError) as e_info:
        targets = utils.map_classes_to(targets, new_classes)
    
