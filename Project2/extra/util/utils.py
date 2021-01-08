import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def swish(x):
    return x*torch.sigmoid(x)


def minmaxnorm(x):
    return (x-x.min())/(x.max()-x.min())

class norm():
    def __init__(self,x):
        self.max = max(x)
        self.min = min(x)
        self.data = (x-self.min)/(self.max-self.min)
        return
    def expand(self,x):
        return x*(self.max()-self.min())+self.min()

def get_class_weights(labels):
    return compute_class_weight("balanced", classes=np.unique(labels), y=labels)


class DataSet(Dataset):
    def __init__(self, samples, labels, m=None, weights=None):
        'Initialization'
        self.labels = labels
        self.samples = samples
        self.m = m
        if len(samples) != len(labels):
            raise ValueError(
                f"should have the same number of samples({len(samples)}) as there are labels({len(labels)})")
        if weights is None:
            self.weights = np.ones_like(labels)
        else:
            if len(weights) != len(labels):
                raise ValueError(
                    f"should have the same number of weights({len(weights)}) as there are samples({len(labels)})")
            self.weights = weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Select sample
        X = self.samples[index]
        y = self.labels[index]
        m = self.m[index] if self.m is not None else self.m
        w = self.weights[index]
        return X, y, m, w


def find_threshold(L, mask, x_frac):
    """
    Calculate c such that x_frac of the array is less than c. Used to calcuate the cut for a given signal efficiency for example.

    Parameters
    ----------
    L : Array
        The array where the cutoff is to be found
    mask : Array,
        Mask that returns L[mask] the part of the original array over which it is desired to calculate the threshold.
    x_frac : float
        Of the area that is lass than or equal to c.

    returns c (type=L.dtype)
    """
    max_x = mask.sum()
    x = int(np.round(x_frac * max_x))
    L_sorted = np.sort(L[mask.astype(bool)])
    return L_sorted[x]
