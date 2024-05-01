import numpy as np
import pytest

from .. import utils

@pytest.mark.parametrize("targets, new_classes", [
    ([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [-1.0, 1.0]),
    ([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [1.0, 2.0]),
    ([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0], [1.0, 4.0]), 
    ([0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0], [-1.0, 1.0, 3.0])
])    
def test_dataset_map_classes(targets, new_classes):
    targets = np.asarray(targets)
    assert np.array_equal(np.unique(utils.map_classes_to(targets, new_classes)), new_classes)
    
def test_dataset_map_classes_error():
    targets = np.asarray([0.0, 1.0, 2.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0])
    new_classes = [-1.0, 1.0]
    with pytest.raises(AssertionError) as e_info:
        targets = utils.map_classes_to(targets, new_classes)
    
