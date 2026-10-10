"""Render the research-log visuals: animated idea map (MP4 + GIF), result charts (PNG),
and a LinkedIn carousel (PDF). Run: python tools/make_assets.py  (needs Pillow, matplotlib, ffmpeg)."""
import subprocess
import textwrap
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
FONTS = Path("C:/Windows/Fonts")
W = H = 1080
FPS = 30

# Dark reference palette (validated: tools/../README "Palette")
BG, CARD = (26, 26, 25), (36, 36, 34)
INK, INK2, MUTED, LINE = (240, 239, 236), (195, 194, 183), (137, 135, 129), (70, 70, 67)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = (57, 135, 229), (217, 89, 38), (25, 158, 112), (201, 133, 0), (144, 133, 233)
GRAY_BAR = (98, 97, 93)
SIGN = "Eric Lam  ·  AI Developer, Toronto  ·  github.com/LamShiuChing"


@lru_cache(None)
def font(size, bold=False):
    return ImageFont.truetype(str(FONTS / ("segoeuib.ttf" if bold else "segoeui.ttf")), size)


def ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def mix(c, a):
    return tuple(round(BG[i] + (c[i] - BG[i]) * a) for i in range(3))


def wrap(txt, size, maxw, bold=False):
    words, lines, cur = txt.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if font(size, bold).getlength(trial) <= maxw:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


class Ctx:
    """Draw helpers for one frame of one scene. t = seconds into scene, k = scene fade (0..1)."""

    def __init__(self, d, t, k):
        self.d, self.t, self.k = d, t, k

    def p(self, t0, dur=0.5):
        return ease((self.t - t0) / dur)

    def text(self, xy, s, size, color=INK, t0=0.0, bold=False, anchor="la", rise=22, maxw=None, gap=1.3):
        p = self.p(t0)
        a = p * self.k
        if a <= 0:
            return
        x, y = xy
        y += (1 - p) * rise
        for i, line in enumerate(wrap(s, size, maxw, bold) if maxw else [s]):
            self.d.text((x, y + i * size * gap), line, font=font(size, bold), fill=mix(color, a), anchor=anchor)

    def card(self, box, t0, stripe=None, outline=None, slide=40):
        p = self.p(t0)
        a = p * self.k
        if a <= 0:
            return 0
        x0, y0, x1, y1 = box
        dx = (1 - p) * -slide
        self.d.rounded_rectangle((x0 + dx, y0, x1 + dx, y1), 18, fill=mix(CARD, a),
                                 outline=mix(outline, a) if outline else None, width=3)
        if stripe:
            self.d.rounded_rectangle((x0 + dx, y0, x0 + dx + 10, y1), 5, fill=mix(stripe, a))
        return dx


# ---------------------------------------------------------------- scenes
def s_question(c):
    c.text((80, 90), "AI RESEARCH LOG  ·  AUG – OCT 2026", 28, BLUE, 0.0, bold=True)
    c.text((80, 200), "Can a model keep", 84, INK, 0.3, bold=True)
    c.text((80, 300), "learning after training,", 84, INK, 0.5, bold=True)
    c.text((80, 400), "without forgetting?", 84, BLUE, 0.7, bold=True)
    c.text((80, 560), "Today's LLMs learn once, then freeze:", 34, INK2, 1.3)
    steps = [("Pretrain", INK), ("Freeze weights", INK), ("Chat ends,\nlesson is lost", ORANGE)]
    for i, (label, col) in enumerate(steps):
        x0 = 80 + i * 320
        dx = c.card((x0, 630, x0 + 280, 790), 1.7 + i * 0.4, outline=col)
        if dx is not None:
            c.text((x0 + 140 + dx, 710 - (17 if "\n" in label else 0)), label.split("\n")[0], 32, col, 1.7 + i * 0.4,
                   bold=True, anchor="mm", rise=0)
            if "\n" in label:
                c.text((x0 + 140 + dx, 727), label.split("\n")[1], 32, col, 1.7 + i * 0.4, bold=True, anchor="mm", rise=0)
        if i < 2:
            c.text((x0 + 300, 710), "→", 40, MUTED, 1.9 + i * 0.4, anchor="mm", rise=0)
    c.text((80, 880), "I spent two months testing ways around that.", 34, INK, 3.0)
    c.text((80, 990), SIGN, 24, MUTED, 0.2)


def s_idea(c):
    import math
    c.text((80, 80), "My idea: learn like a brain", 60, INK, 0.0, bold=True)
    c.text((80, 165), "Fast memory for today. Slow weights for life. Sleep in between.", 30, INK2, 0.3)
    cx, cy, r = 540, 610, 200
    arc_p = c.p(2.0, 1.2)
    if arc_p > 0:
        c.d.arc((cx - r, cy - r, cx + r, cy + r), 270, 270 + 360 * arc_p, fill=mix(LINE, c.k), width=5)
    if c.t > 3.0:  # a thought travelling round the loop
        ang = math.radians(270 + 360 * (((c.t - 3.0) / 2.6) % 1))
        x, y = cx + r * math.cos(ang), cy + r * math.sin(ang)
        c.d.ellipse((x - 18, y - 18, x + 18, y + 18), fill=mix(BLUE, 0.35 * c.k))
        c.d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=mix(INK, c.k))
    nodes = [  # angle, title, sub, colour, anchor, title offset, sub offset (from node centre)
        (270, "Experience", "new data, new tasks", BLUE, "ms", (0, -66), (0, -32)),
        (0, "Fast memory", "\"hippocampus\", 1-s writes", AQUA, "ls", (36, -6), (36, 26)),
        (90, "Sleep & replay", "consolidate, don't overwrite", VIOLET, "ms", (0, 72), (0, 104)),
        (180, "Long-term weights", "\"cortex\": slow, shared", YELLOW, "rs", (-36, -6), (-36, 26)),
    ]
    for i, (deg, title, sub, col, anc, (tx, ty), (sx, sy)) in enumerate(nodes):
        t0 = 0.6 + i * 0.35
        a = c.p(t0) * c.k
        if a <= 0:
            continue
        x, y = cx + r * math.cos(math.radians(deg)), cy + r * math.sin(math.radians(deg))
        rr = 16 + 6 * (1 - c.p(t0))
        c.d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=mix(col, a))
        c.text((x + tx, y + ty), title, 32, col, t0, bold=True, anchor=anc, rise=0)
        c.text((x + sx, y + sy), sub, 23, INK2, t0, anchor=anc, rise=0)
    c.text((cx, cy - 22), "learn fast,", 36, INK, 2.4, bold=True, anchor="mm", rise=0)
    c.text((cx, cy + 26), "consolidate slow", 36, INK, 2.6, bold=True, anchor="mm", rise=0)
    c.text((80, 1000), "Inspired by complementary learning systems theory (hippocampus + cortex)", 22, MUTED, 1.0)


CARDS = [
    ("Memory", AQUA, "Fact slots on a frozen transformer · least-squares memory · sleep replay"),
    ("Continual learning", BLUE, "Re-tested Dohare et al. (Nature 2024) on loss of plasticity · 13 claims"),
    ("Reasoning from scratch", YELLOW, "Trained GPTs 5M–124M · length extrapolation · composition · proofs"),
    ("Agents & tools", VIOLET, "LoRA-tuned Qwen3-0.6B tool agent · Qwen2.5-14B memory gate · ARC-AGI-3"),
    ("Brain-inspired wiring", ORANGE, "Real fruit-fly connectome vs RNN/GRU · DreamCoder-style program learning"),
]


def s_tested(c):
    c.text((80, 80), "What I built & tested", 60, INK, 0.0, bold=True)
    c.text((80, 165), "20+ experiments  ·  1 GPU (RTX 4080 SUPER)  ·  ~$5 of cloud", 30, INK2, 0.3)
    for i, (title, col, desc) in enumerate(CARDS):
        y0, t0 = 270 + i * 150, 0.6 + i * 0.35
        dx = c.card((80, y0, 1000, y0 + 128), t0, stripe=col)
        if dx is None:
            continue
        c.text((118 + dx, y0 + 22), title, 34, col, t0, bold=True, rise=0)
        c.text((118 + dx, y0 + 74), desc, 23, INK2, t0, rise=0)
    c.text((80, 1030), "Every run: pass/fail bar written first · controls · multiple seeds", 23, MUTED, 2.6)


TILES = [  # big number formatter(progress), colour, title, subtitle
    (lambda p: f"{max(1, round(7 * p))}×", BLUE, "less forgetting", "GPT sleeps on its own dreams: 0.58 → 0.08 bits/byte"),
    (lambda p: f"{round(2000 * p):,}", AQUA, "old facts kept at 100%", "while new facts are written in ~1 s each"),
    (lambda p: f"{0.74 * p:.2f}", YELLOW, "length extrapolation", "rotary positions: 0.00 → no positions + per-doc attention"),
    (lambda p: f"{0.81 * p:.2f}", VIOLET, "unseen compositions", "up from ≤ 0.15 by adding data diversity, not size"),
]


def s_results(c):
    c.text((80, 80), "Results that held up", 60, INK, 0.0, bold=True)
    c.text((80, 165), "Small models, careful tests, numbers from my own runs", 30, INK2, 0.3)
    for i, (num, col, title, sub) in enumerate(TILES):
        x0, y0 = 80 + (i % 2) * 470, 260 + (i // 2) * 330
        t0 = 0.5 + i * 0.4
        dx = c.card((x0, y0, x0 + 450, y0 + 305), t0, stripe=col)
        if dx is None:
            continue
        c.text((x0 + 40 + dx, y0 + 30), num(c.p(t0 + 0.2, 1.2)), 92, col, t0, bold=True, rise=0)
        c.text((x0 + 40 + dx, y0 + 160), title, 30, INK, t0, bold=True, rise=0)
        c.text((x0 + 40 + dx, y0 + 208), sub, 22, INK2, t0, rise=0, maxw=380)
    c.text((80, 945), "Failures are logged too: real fly wiring lost to a GRU; a 14B model deciding when to "
                      "retrieve used 64–76% more tokens than a simple rule.", 23, MUTED, 2.4, maxw=920)


def s_learned(c):
    c.text((80, 80), "What I learned", 60, INK, 0.0, bold=True)
    lessons = [
        "Simple baselines beat clever mechanisms more often than you'd think. Better to learn that in a day.",
        "Generalization came from data diversity and outside verification, not model size or self-checking.",
        "Dense shared weights forget. Replay, sleep and regularization only partly fix it.",
    ]
    for i, s in enumerate(lessons):
        y, t0 = 210 + i * 175, 0.4 + i * 0.5
        c.text((80, y - 8), f"0{i + 1}", 54, BLUE, t0, bold=True)
        c.text((200, y), s, 32, INK, t0, maxw=800)
    dx = c.card((80, 760, 1000, 920), 2.2, outline=BLUE)
    if dx is not None:
        c.text((118 + dx, 785), "NEXT", 24, BLUE, 2.2, bold=True, rise=0)
        c.text((118 + dx, 822), "Architectures where new learning doesn't overwrite old knowledge, running on "
                                "a local device.", 30, INK, 2.2, bold=True, rise=0, maxw=840)
    c.text((80, 990), SIGN, 24, INK2, 2.8)


SCENES = [(s_question, 5.0), (s_idea, 6.5), (s_tested, 5.5), (s_results, 6.5), (s_learned, 7.0)]
TOTAL = sum(d for _, d in SCENES)


def frame(t, fade=True):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    start = 0.0
    for fn, dur in SCENES:
        if start <= t < start + dur:
            lt = t - start
            k = ease((dur - lt) / 0.45) if fade else 1.0
            fn(Ctx(d, lt, k))
            break
        start += dur
    if fade:  # progress bar on the video only, not on carousel stills
        d.rectangle((0, H - 6, W * t / TOTAL, H), fill=BLUE)
    return img


def render_video():
    mp4, gif = OUT / "idea-map.mp4", OUT / "idea-map.gif"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "18", "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    for i in range(int(TOTAL * FPS)):
        ff.stdin.write(frame(i / FPS).tobytes())
    ff.stdin.close()
    ff.wait()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=none",
                    str(gif)], check=True)
    snaps = []
    start = 0.0
    for _, dur in SCENES:  # fully-built end state of each scene, for the carousel
        snaps.append(frame(start + dur - 0.05, fade=False))
        start += dur
    return snaps


# ---------------------------------------------------------------- static charts
def hexc(c):
    return "#%02x%02x%02x" % c


plt.rcParams.update({"font.family": "Segoe UI", "text.color": hexc(INK)})


def chart(name, title, subtitle, rows, fmt, note):
    """Horizontal comparison bars. rows: (label, value, is_mine). Blue = what I built, gray = baseline."""
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=hexc(BG))
    fig.text(0.074, 0.905, title, fontsize=34, weight="bold")
    fig.text(0.074, 0.845, "\n".join(textwrap.wrap(subtitle, 72)), fontsize=19, color=hexc(INK2), va="top")
    ax = fig.add_axes([0.074, 0.27, 0.85, 0.46], facecolor=hexc(BG))
    vmax = max(v for _, v, _ in rows)
    n = len(rows)
    for i, (label, v, mine) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, height=0.42, color=hexc(BLUE if mine else GRAY_BAR))
        ax.text(0, y + 0.36, label, fontsize=19, color=hexc(INK if mine else INK2), va="bottom")
        ax.text(v + vmax * 0.02, y, fmt(v), fontsize=26, weight="bold", va="center",
                color=hexc(INK if mine else INK2))
    ax.set_xlim(0, vmax * 1.22)
    ax.set_ylim(-0.6, n - 0.2)
    ax.axvline(0, color=hexc(LINE), lw=2)
    ax.axis("off")
    fig.text(0.074, 0.2, "\n".join(textwrap.wrap(note, 78)), fontsize=17, color=hexc(MUTED), va="top")
    fig.text(0.074, 0.06, SIGN, fontsize=15, color=hexc(MUTED))
    path = OUT / f"{name}.png"
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def claims_chart():
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=hexc(BG))
    fig.text(0.074, 0.905, "Re-testing a Nature paper myself", fontsize=34, weight="bold")
    fig.text(0.074, 0.845, "\n".join(textwrap.wrap(
        "13 claims from Dohare et al., \"Loss of plasticity in deep continual learning\" (Nature 2024) "
        "and a Rich Sutton interview. Permuted MNIST, 3 seeds, falsifier written before each run.", 62)),
        fontsize=19, color=hexc(INK2), va="top")
    ax = fig.add_axes([0.074, 0.5, 0.85, 0.12], facecolor=hexc(BG))
    ax.barh(0, 7, height=0.6, color=hexc(BLUE))
    ax.barh(0, 6, left=7.06, height=0.6, color=hexc(ORANGE))  # small surface gap between segments
    ax.text(3.5, 0, "7 held up", fontsize=26, weight="bold", ha="center", va="center", color=hexc(INK))
    ax.text(10.06, 0, "6 did not", fontsize=26, weight="bold", ha="center", va="center", color=hexc(INK))
    ax.set_xlim(0, 13.06)
    ax.axis("off")
    held = ["Plain backprop loses plasticity", "Adam is worse", "L2 helps but doesn't eliminate it (−2.4 to −3.0 pts)",
            "Replay reduces forgetting"]
    fell = ["Shrink-and-perturb nearly eliminates the loss", "\"Forgetting is totally curable\" with continual backprop",
            "The three correlates all move as described"]
    fig.text(0.074, 0.44, "Held up, e.g.", fontsize=20, weight="bold", color=hexc(BLUE))
    fig.text(0.074, 0.41, "\n".join("·  " + s for s in held), fontsize=17, color=hexc(INK2), va="top", linespacing=1.6)
    fig.text(0.074, 0.25, "Did not hold in my setup, e.g.", fontsize=20, weight="bold", color=hexc(ORANGE))
    fig.text(0.074, 0.22, "\n".join("·  " + s for s in fell), fontsize=17, color=hexc(INK2), va="top", linespacing=1.6)
    fig.text(0.074, 0.06, SIGN, fontsize=15, color=hexc(MUTED))
    path = OUT / "result-plasticity-claims.png"
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def render_charts():
    f2 = lambda v: f"{v:.2f}"
    return [
        chart("result-sleep-forgetting", "Sleep cut forgetting 7×",
              "A small GPT learns day by day. Between days it 'sleeps': rehearses and replays its own generated text.",
              [("Normal day-by-day training", 0.58, False), ("+ sleep on its own dreams", 0.08, True)], f2,
              "Forgetting in bits per byte, lower is better. Much less forgetting, but not zero."),
        chart("result-cifar-retention", "Remembering day 1 after 20 days",
              "CIFAR-100 learned as a 20-day stream, frozen image encoder. Accuracy on day-1 classes at the end.",
              [("Normal network", 0.00, False), ("Running-sum link memory", 0.79, True)], f2,
              "The link memory also matched learning everything at once on all classes (0.60 vs 0.60)."),
        chart("result-fact-memory", "Memory slots on a frozen transformer",
              "Write a new fact in ~25 steps (~1 s) without touching the backbone, then answer 2-hop questions.",
              [("Single forward pass", 0.010, False), ("Memory slots + symbolic workspace", 1.000, True)],
              lambda v: f"{v:.3f}",
              "2-hop accuracy. 2,000 earlier facts stayed at 1.000, zero drift over 100 writes. Synthetic, closed "
              "vocabulary: unseen words still fail."),
        chart("result-length-extrapolation", "Counting past what it was trained on",
              "Small GPTs trained from scratch in a mixed prose + reasoning stream. Test: 6-digit addition, longer than training.",
              [("Rotary positions (5M to 85M)", 0.00, False), ("No positions + per-document attention", 0.74, True)],
              f2, "Scaling 5M → 85M left rotary at exactly 0.00. Removing positions and masking attention per document fixed it."),
        chart("result-composition", "Composition is a data problem",
              "Can a 5M model chain two skills it never saw chained, when asked in English?",
              [("Few combinations trained", 0.15, False), ("34 of 42 combinations trained", 0.81, True)], f2,
              "Accuracy on held-out pairs (low-diversity run: 0–15 of 100). More model size didn't help. More variety did."),
        chart("result-memory-gate", "A 14B model as memory gate",
              "Qwen2.5-14B (local) as a retrieve-or-act gate vs simple policies, same tasks.",
              [("Simple heuristic", 7631, False), ("Always retrieve", 8185, False), ("Qwen2.5-14B decides", 13439, True)],
              lambda v: f"{v:,.0f}",
              "Total tokens, lower is better. The model gate also skipped two retrievals it needed. A negative result, logged as one."),
        claims_chart(),
    ]


def render_carousel(snaps, charts):
    order = [snaps[0], snaps[1], snaps[2], *[Image.open(p).convert("RGB") for p in charts], snaps[3], snaps[4]]
    order[0].save(OUT / "linkedin-carousel.pdf", save_all=True, append_images=order[1:], resolution=100)
    for i, s in enumerate(snaps):
        s.save(OUT / f"slide-{i + 1}.png")


if __name__ == "__main__":
    charts = render_charts()
    snaps = render_video()
    render_carousel(snaps, charts)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:40s} {p.stat().st_size / 1e6:6.2f} MB")
