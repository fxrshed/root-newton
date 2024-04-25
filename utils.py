import os
import numpy as np

from sklearn.datasets import load_svmlight_file

from dotenv import load_dotenv
load_dotenv()

def get_LIBSVM(dataset_name: str):
    datasets_path = os.getenv("LIBSVM_DIR")
    trainX, trainY = load_svmlight_file(f"{datasets_path}/{dataset_name}")
    return trainX.toarray(), trainY

def make_synthetic_binary_classification(n_samples: int, n_features: int, symmetric: bool = False, seed: int = 0):
    np.random.seed(seed)
    data = np.random.randn(n_samples, n_features)
    
    if symmetric:
        assert n_samples == n_features, f"n_samples must be equal to n_features to get symmetric matrix. " \
            f"Currently n_samples={n_samples}, n_features={n_features}."
        data = (data + data.T) / 2
    w_star = np.random.randn(n_features)

    target = data @ w_star
    target[target <= 0.0] = -1.0
    target[target > 0.0] = 1.0

    return data, target


def map_classes_to(target, new_classes):
    old_classes = np.unique(target)
    new_classes = np.sort(new_classes)
    
    if np.array_equal(old_classes, new_classes):
        return target
    
    assert np.unique(target).size == len(new_classes), \
        f"Old classes must match the number of new classes. " \
        f"Currently ({np.unique(target).size}) classes are being mapped to ({len(new_classes)}) new classes."

    mapping = {v: t for v, t in zip(old_classes, new_classes)}
    target = np.vectorize(mapping.get)(target)
    return target