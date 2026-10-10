"""Continual backpropagation (Dohare et al., Nature 2024): generate-and-test on hidden units.

After every gradient step, each hidden unit's utility is an exponential average of
|activation| * sum of |outgoing weights| (paper eq. 1). Among units older than the maturity
threshold, a fraction `rate` of the least useful is reinitialised: incoming weights redrawn from
the init distribution, bias and outgoing weights set to zero so the network function is unchanged.
Deviation from the reference code (lop/algos/gnt.py): no downstream-bias compensation.
"""
import math

import torch


class ContinualBackprop:
    def __init__(self, net, rate=1e-4, maturity=100, decay=0.99, random=False):
        self.layers = [*net.hidden, net.out]
        self.rate, self.maturity, self.decay, self.random = rate, maturity, decay, random
        dev = net.out.weight.device
        self.util = [torch.zeros(l.out_features, device=dev) for l in net.hidden]
        self.age = [torch.zeros(l.out_features, device=dev) for l in net.hidden]
        self.owed = [0.0] * len(net.hidden)  # fractional replacements carried between steps

    @torch.no_grad()
    def step(self, feats):
        for i, h in enumerate(feats):
            self.age[i] += 1
            outgoing = self.layers[i + 1].weight.abs().sum(0)
            self.util[i].mul_(self.decay).add_(h.abs().mean(0) * outgoing, alpha=1 - self.decay)
            eligible = (self.age[i] > self.maturity).nonzero().squeeze(1)
            self.owed[i] += self.rate * len(eligible)
            n = int(self.owed[i])
            if n == 0:
                continue
            self.owed[i] -= n
            if self.random:  # control arm (C11): same churn, no utility
                pick = torch.randperm(len(eligible), device=eligible.device)[:n]
            else:
                corrected = self.util[i][eligible] / (1 - self.decay ** self.age[i][eligible])
                pick = corrected.topk(n, largest=False).indices
            self.reinit(i, eligible[pick])

    def reinit(self, i, idx):
        lin, nxt = self.layers[i], self.layers[i + 1]
        bound = 1 / math.sqrt(lin.in_features)  # nn.Linear's default uniform bound
        lin.weight[idx] = torch.empty(len(idx), lin.in_features, device=idx.device).uniform_(-bound, bound)
        lin.bias[idx] = 0
        nxt.weight[:, idx] = 0
        self.util[i][idx] = 0
        self.age[i][idx] = 0
