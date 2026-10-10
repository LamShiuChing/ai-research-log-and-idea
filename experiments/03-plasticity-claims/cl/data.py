"""MNIST held on the device as flat tensors; permuted-MNIST task streams."""
import torch
from torchvision.datasets import MNIST

ROOT = "data"
DIM = 784


def load(device):
    """(x_train, y_train, x_test, y_test): pixels in [0, 1], images flattened to 784."""
    sets = [MNIST(ROOT, train=train, download=True) for train in (True, False)]
    return tuple(t.to(device) for s in sets for t in (s.data.float().div(255).flatten(1), s.targets))


def permutations(n_tasks, gen, device):
    return [torch.randperm(DIM, generator=gen).to(device) for _ in range(n_tasks)]


def batches(x, y, perm, n_examples, batch, gen):
    """One task's online stream: n_examples drawn without replacement, each seen once."""
    order = torch.randperm(len(x), generator=gen)[:n_examples].to(x.device)
    for i in range(0, n_examples, batch):
        idx = order[i:i + batch]
        yield x[idx][:, perm], y[idx]
