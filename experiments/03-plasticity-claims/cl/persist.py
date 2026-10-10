"""Byte-exact save/restore of everything that trains: model, optimizer, CBP state, reservoir, and
all three RNG streams (global CPU, global CUDA, the run's Generator). checks/persistence.py is the
proof (HOLES E19); sleep experiments build on this."""
import torch


def save(path, learner, gen):
    cbp, res = learner.cbp, learner.replay
    torch.save(dict(
        net=learner.net.state_dict(),
        opt=learner.opt.state_dict(),
        cbp=dict(util=cbp.util, age=cbp.age, owed=cbp.owed) if cbp else None,
        replay=dict(x=res.x, y=res.y, seen=res.seen) if res else None,
        rng=dict(cpu=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                 gen=gen.get_state()),
    ), path)


def save_lm(path, net, opt, res, gen):
    """Tier-2 stack: GPT, AdamW, TokenReservoir, all three RNG streams."""
    torch.save(dict(
        net=net.state_dict(), opt=opt.state_dict(),
        res=dict(x=res.x, seen=res.seen) if res else None,
        rng=dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), gen=gen.get_state()),
    ), path)


def load_lm(path, net, opt, res, gen):
    s = torch.load(path, weights_only=True)
    net.load_state_dict(s["net"])
    opt.load_state_dict(s["opt"])
    if s["res"]:
        res.x, res.seen = s["res"]["x"], s["res"]["seen"]
    torch.set_rng_state(s["rng"]["cpu"])
    torch.cuda.set_rng_state(s["rng"]["cuda"])
    gen.set_state(s["rng"]["gen"])


def load(path, learner, gen):
    s = torch.load(path, weights_only=True)
    learner.net.load_state_dict(s["net"])
    learner.opt.load_state_dict(s["opt"])
    if s["cbp"]:
        learner.cbp.util, learner.cbp.age, learner.cbp.owed = s["cbp"]["util"], s["cbp"]["age"], s["cbp"]["owed"]
    if s["replay"]:
        learner.replay.x, learner.replay.y, learner.replay.seen = s["replay"]["x"], s["replay"]["y"], s["replay"]["seen"]
    torch.set_rng_state(s["rng"]["cpu"])
    if s["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state(s["rng"]["cuda"])
    gen.set_state(s["rng"]["gen"])
