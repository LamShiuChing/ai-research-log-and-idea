"""One online update per call. Variants compose: L2 (weight decay), shrink-and-perturb (weight
decay + post-step Gaussian noise), continual backprop, and experience replay."""
import torch
import torch.nn.functional as F


class Reservoir:
    """Reservoir-sampled memory of past examples."""

    def __init__(self, size, dim, device):
        self.x = torch.empty(size, dim, device=device)
        self.y = torch.empty(size, dtype=torch.long, device=device)
        self.seen = 0

    def add(self, x, y):
        for xi, yi in zip(x, y):
            j = self.seen if self.seen < len(self.x) else torch.randint(self.seen + 1, ()).item()
            if j < len(self.x):
                self.x[j], self.y[j] = xi, yi
            self.seen += 1

    def sample(self, n):
        idx = torch.randint(min(self.seen, len(self.x)), (n,), device=self.x.device)
        return self.x[idx], self.y[idx]


class Learner:
    def __init__(self, net, lr, adam=False, weight_decay=0.0, noise=0.0, cbp=None, replay=None):
        self.net, self.noise, self.cbp, self.replay = net, noise, cbp, replay
        opt = torch.optim.Adam if adam else torch.optim.SGD
        self.opt = opt(net.parameters(), lr=lr, weight_decay=weight_decay)

    def step(self, x, y, store=True):
        """Update on (x, y); returns the pre-update logits for x (online accuracy is scored on these).
        store=False trains without adding to the reservoir (C15's never-rehearsed probe domain)."""
        n = len(y)
        if self.replay and self.replay.seen:
            rx, ry = self.replay.sample(n)
            x, y = torch.cat([x, rx]), torch.cat([y, ry])
        logits, feats = self.net(x)
        loss = F.cross_entropy(logits, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            if self.noise:
                for p in self.net.parameters():
                    p.add_(torch.randn_like(p), alpha=self.noise)
            if self.cbp:
                self.cbp.step(feats)
            if self.replay and store:
                self.replay.add(x[:n], y[:n])
        return logits[:n].detach()
