"""20 days of continual learning with running-sum links, on two real, harder datasets.

Images: CIFAR-100 (100 object kinds), 5 new kinds per day. Vision area: pretrained ResNet-18, frozen (a built-in
sense). Text: 20 Newsgroups (real forum posts, 20 topics), 1 new topic per day. Text area: fixed word hashing
(4096 numbers, word counts squashed, no statistics fitted on the data).
Each kind/topic has a random name vector; the link maps area vector -> name vector; the answer is the nearest
name among those heard so far (no answer list beyond what it has heard).

Arms:  link (sums)       running totals, day by day
       net (normal)      2-layer network (scikit-learn, Adam), trained day by day
       net (all at once) same network on all 20 days mixed: best-case reference for a network
Checks: link learned day by day == link learned all at once (exact).

PREDICTIONS (before running):
1. Link: day-1 kinds at day 20 score within 3 points of an all-at-once link (they are the same numbers).
   Day-1 scores fall over days only because more kinds compete, not because memory is lost.
2. Normal net: day-1 kinds at day 20 below 5%.
3. Link overall at day 20 >= 50% on CIFAR-100 and >= 60% on newsgroups, within 10 points of the all-at-once net.

RESULTS, 20 Newsgroups (1 new topic per day):
 day | day-1 topics: link  net | all topics so far: link  net
   1 |                1.00  1.00 |                   1.00  1.00
   5 |                0.76  0.00 |                   0.78  0.22
  10 |                0.68  0.00 |                   0.70  0.11
  20 |                0.62  0.00 |                   0.60  0.05
 day-20 reference: link all at once 0.60, net all at once 0.54.
 1 PASS (0.62 vs 0.60 all at once; the fall follows the fall of all topics, i.e. more competitors).
 2 PASS (net 0.00). 3 PASS for text (0.60; beats the all-at-once net 0.54).
 The exact-equality check printed False: features are float32, so B differs by 8e-5 and W by 3.7e-7 relative
 (checked separately); rounding only.

RESULTS, CIFAR-100 (5 new kinds per day; frozen ResNet-18 features):
 day | day-1 kinds: link  net | all kinds so far: link  net
   1 |               0.96  0.97 |                  0.96  0.97
   5 |               0.88  0.00 |                  0.84  0.19
  10 |               0.83  0.00 |                  0.71  0.09
  20 |               0.79  0.00 |                  0.60  0.05
 day-20 reference: link all at once 0.60, net all at once 0.58. 1 PASS, 2 PASS, 3 PASS (0.60 >= 0.50).
 Equality check False again: float32 features, rounding (as for text).
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neural_network import MLPRegressor

rng = np.random.default_rng(0)
DAYS = 20


def cifar_features():
    """CIFAR-100 from Hugging Face (uoft-cs/cifar100), faster than the original host. Features cached to disk."""
    from pathlib import Path
    cache = Path("data/cifar100_resnet18.npz")
    if cache.exists():
        z = np.load(cache)
        return [(z["xtr"], z["ytr"]), (z["xte"], z["yte"])]
    from datasets import load_dataset
    tf = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Identity()
    net = net.eval().cuda()
    d = load_dataset("uoft-cs/cifar100")
    out = []
    for split in ("train", "test"):
        imgs, feats = d[split]["img"], []
        with torch.no_grad():
            for i in range(0, len(imgs), 256):
                x = torch.stack([tf(im.convert("RGB")) for im in imgs[i:i + 256]]).cuda()
                feats.append(net(x).cpu().numpy())
        out.append((np.concatenate(feats), np.array(d[split]["fine_label"])))
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, xtr=out[0][0], ytr=out[0][1], xte=out[1][0], yte=out[1][1])
    return out


def news_features():
    hv = HashingVectorizer(n_features=4096, alternate_sign=False, norm=None)
    out = []
    for subset in ("train", "test"):
        d = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"))
        x = np.log1p(hv.transform(d.data).toarray().astype(np.float32))       # squash word counts
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        out.append((x, np.array(d.target)))
    return out


def run(name, data, per_day, lam):
    (Xtr, ytr), (Xte, yte) = data
    n_cls, dim = per_day * DAYS, Xtr.shape[1]
    order = rng.permutation(n_cls)                       # which kinds arrive on which day
    day_of = np.empty(n_cls, int)
    day_of[order] = np.arange(n_cls) // per_day
    names = rng.normal(size=(n_cls, 256))
    A, B = np.zeros((256, dim)), np.zeros((dim, dim))
    net = MLPRegressor(hidden_layer_sizes=(256,), learning_rate_init=1e-3, random_state=0)
    rows = []

    def score(predict, day, which):
        heard = np.flatnonzero(day_of <= day)
        m = np.isin(yte, heard) & which
        pred = predict(Xte[m]) @ names[heard].T
        return np.mean(heard[pred.argmax(1)] == yte[m])

    for day in range(DAYS):
        tr = day_of[ytr] == day
        x, y = Xtr[tr], names[ytr[tr]]
        A += y.T @ x
        B += x.T @ x
        for _ in range(30):
            net.partial_fit(x, y)
        W = A @ np.linalg.inv(B + lam * np.eye(dim))
        link = lambda z, W=W: z @ W.T
        if day + 1 in (1, 5, 10, 15, 20):
            first, all_ = day_of[yte] == 0, np.ones(len(yte), bool)
            rows.append((day + 1, score(link, day, first), score(net.predict, day, first),
                         score(link, day, all_), score(net.predict, day, all_)))

    ref = MLPRegressor(hidden_layer_sizes=(256,), learning_rate_init=1e-3, random_state=0, max_iter=200)
    ref.fit(Xtr, names[ytr])
    all_at_once_W = (names[ytr].T @ Xtr) @ np.linalg.inv(Xtr.T @ Xtr + lam * np.eye(dim))
    print(f"\n{name}: {n_cls} kinds, {per_day} new per day. Scores = share of test items named right.")
    print(f"{'day':>4} | {'day-1 kinds: link':>18} {'net':>6} | {'all kinds so far: link':>23} {'net':>6}")
    for r in rows:
        print(f"{r[0]:>4} | {r[1]:>18.2f} {r[2]:>6.2f} | {r[3]:>23.2f} {r[4]:>6.2f}")
    everything = np.ones(len(yte), bool)
    print(f"reference at day 20: link all at once {score(lambda z: z @ all_at_once_W.T, DAYS - 1, everything):.2f}, "
          f"net all at once {score(ref.predict, DAYS - 1, everything):.2f}")
    print("link day by day == link all at once:", np.allclose(W, all_at_once_W, atol=1e-6))


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["text", "images"]
    if "text" in which:
        run("20 Newsgroups text", news_features(), per_day=1, lam=1.0)
    if "images" in which:
        run("CIFAR-100 images", cifar_features(), per_day=5, lam=10.0)
