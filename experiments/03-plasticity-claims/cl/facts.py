"""The facts world (F1 made operational): N random key->value associations, keys fixed random
vectors, values fixed random classes. The stream draws facts iid from a Zipf law over the N
(or uniformly — the no-structure control). Relearning is expensive by scarcity: a forgotten fact
cannot be relearned until the stream shows it again, and most facts are cold."""
import torch


def world(n, gen, device):
    keys = torch.randn(n, 784, generator=gen).to(device)
    values = torch.randint(10, (n,), generator=gen).to(device)
    return keys, values


def stream(n, total, s, gen):
    """Fact indices for the whole life: Zipf(s) over ranks (s=0 -> uniform)."""
    w = 1 / torch.arange(1, n + 1).float() ** s
    return torch.multinomial(w, total, replacement=True, generator=gen)
