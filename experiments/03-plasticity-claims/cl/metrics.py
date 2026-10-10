import torch


@torch.no_grad()
def accuracy(net, x, y):
    return (net(x)[0].argmax(1) == y).float().mean().item()


@torch.no_grad()
def diagnostics(net, x):
    """Plasticity correlates from the paper: dead-unit fraction, mean |weight|, effective rank of the
    last hidden layer (exp of the entropy of its normalised singular values)."""
    _, feats = net(x)
    alive = torch.cat([(f > 0).any(0) for f in feats])
    weights = torch.cat([p.abs().flatten() for name, p in net.named_parameters() if name.endswith("weight")])
    p = torch.linalg.svdvals(feats[-1])
    p = p / p.sum()
    erank = torch.exp(-(p * p.clamp_min(1e-12).log()).sum())
    return dict(dead=1 - alive.float().mean().item(), wmag=weights.mean().item(), erank=erank.item())


def forgetting(acc):
    """Mean over past tasks of (best accuracy ever - final accuracy). acc[t][j] = test accuracy on
    task j after training task t."""
    last = len(acc) - 1
    return sum(max(acc[k][j] for k in range(j, last)) - acc[last][j] for j in range(last)) / last
