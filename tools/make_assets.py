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


# ---------------------------------------------------------------- scenes (vision -> idea -> key finding -> next)
def s_vision(c):
    c.text((80, 120), "MY VISION", 28, BLUE, 0.0, bold=True)
    c.text((80, 230), "AI that keeps learning", 80, INK, 0.3, bold=True)
    c.text((80, 330), "after training,", 80, INK, 0.5, bold=True)
    c.text((80, 430), "on your own device.", 80, BLUE, 0.7, bold=True)
    c.text((80, 640), "Today's models are trained once, then frozen.", 36, INK2, 1.6)
    c.text((80, 700), "What they learn in a conversation is gone when it ends.", 36, INK2, 2.0)
    c.text((80, 990), SIGN, 24, MUTED, 0.2)


STEPS = [  # tag, title, description, colour
    ("DAY", "Fast memory", "writes new experience in seconds", AQUA),
    ("NIGHT", "Sleep", "replays its own dreams to keep what it knew", VIOLET),
    ("FOR LIFE", "Long-term weights", "hold old and new skills together", YELLOW),
]


def s_idea(c):
    c.text((80, 120), "THE IDEA", 28, BLUE, 0.0, bold=True)
    c.text((80, 180), "Learn like a brain", 72, INK, 0.2, bold=True)
    active = int((c.t - 2.4) / 0.9) % 3 if c.t > 2.4 else -1  # cycle highlight through the loop
    for i, (tag, title, desc, col) in enumerate(STEPS):
        x0, t0 = 80 + i * 330, 0.7 + i * 0.45
        dx = c.card((x0, 360, x0 + 290, 700), t0, outline=col if i == active else None, slide=0)
        if dx is None:
            continue
        c.d.rounded_rectangle((x0, 360, x0 + 290, 372), 6, fill=mix(col, c.p(t0) * c.k))
        c.text((x0 + 30, 400), tag, 24, col, t0, bold=True)
        c.text((x0 + 30, 440), title, 36, INK, t0, bold=True, maxw=240, gap=1.15)
        c.text((x0 + 30, 560), desc, 25, INK2, t0, maxw=235)
        if i < 2:
            c.text((x0 + 310, 530), "→", 44, MUTED, t0 + 0.3, anchor="mm", rise=0)
    c.text((80, 790), "No old data is stored. It rehearses by dreaming.", 36, INK, 2.2, bold=True)
    c.text((80, 990), "Inspired by complementary learning systems theory (hippocampus + cortex)", 22, MUTED, 2.6)


def s_finding(c):
    c.text((80, 120), "MOST PROMISING FINDING", 28, BLUE, 0.0, bold=True)
    n = 1 + 6 * c.p(1.6, 2.2)
    c.text((80, 170), f"Up to {n:.0f}× less forgetting", 76, INK, 0.2, bold=True)
    c.text((80, 290), "A small GPT learned 6 kinds of English, one per day. 3 seeds each.", 30, INK2, 0.5)
    rows = [("Normal day-by-day training", 0.58, 0.58, GRAY_BAR),
            ("+ sleep on its own dreams", 0.58, 0.19, BLUE),
            ("+ stronger sleep (more dreams, keep harder)", 0.58, 0.08, BLUE)]
    full = 800  # px for 0.58
    for i, (label, v0, v1, col) in enumerate(rows):
        y, t0 = 380 + i * 150, 0.8 + i * 0.3
        a = c.p(t0) * c.k
        if a <= 0:
            continue
        v = v0 + (v1 - v0) * c.p(1.6, 2.2)
        c.text((80, y), label, 28, INK if col == BLUE else INK2, t0, bold=col == BLUE, rise=0)
        w = full * v / 0.58 * c.p(t0, 0.6)
        c.d.rounded_rectangle((80, y + 46, 80 + max(w, 8), y + 98), 8, fill=mix(col, a))
        c.text((80 + max(w, 8) + 20, y + 72), f"{v:.2f}", 38, INK, t0, bold=True, anchor="lm", rise=0)
    c.text((80, 840), "Forgetting in bits per byte, lower is better.", 26, MUTED, 1.2)
    c.text((80, 885), "No old text is stored. It rehearses by dreaming.", 32, INK, 3.6, bold=True)
    c.text((80, 990), "Byte-level nanoGPT · BabyLM 10M, 6 domains · single RTX 4080 SUPER", 22, MUTED, 1.2)


def s_next(c):
    c.text((80, 120), "NEXT", 28, BLUE, 0.0, bold=True)
    c.text((80, 180), "Make it last.", 80, INK, 0.2, bold=True)
    c.text((80, 330), "Over long streams, shared weights still forget. I'm building architectures where new "
                      "learning doesn't overwrite old knowledge, small enough for a local device.",
           38, INK2, 0.6, maxw=920)
    dx = c.card((80, 640, 1000, 860), 1.6, outline=BLUE)
    if dx is not None:
        c.text((120 + dx, 680), "Eric Lam  ·  AI Developer, Toronto", 34, INK, 1.6, bold=True, rise=0)
        c.text((120 + dx, 740), "github.com/LamShiuChing/ai-research-log-and-idea", 30, BLUE, 1.6, rise=0)
        c.text((120 + dx, 795), "20+ experiments · code · logs · papers", 26, INK2, 1.6, rise=0)


SCENES = [(s_vision, 5.0), (s_idea, 6.5), (s_finding, 7.0), (s_next, 5.5)]
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
    # current per-claim verdicts in CLAIMS.md (C10 was re-graded from falsified to undecided)
    ax.barh(0, 8, height=0.6, color=hexc(BLUE))
    ax.barh(0, 4, left=8.06, height=0.6, color=hexc(ORANGE))  # small surface gap between segments
    ax.barh(0, 1, left=12.12, height=0.6, color=hexc(GRAY_BAR))
    ax.text(4, 0, "8 held up", fontsize=26, weight="bold", ha="center", va="center", color=hexc(INK))
    ax.text(10.06, 0, "4 did not", fontsize=26, weight="bold", ha="center", va="center", color=hexc(INK))
    ax.text(12.62, -0.55, "1 undecided", fontsize=15, ha="center", va="top", color=hexc(INK2))
    ax.set_xlim(0, 13.12)
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
        chart("result-sleep-forgetting", "Sleep cut forgetting 3–7×",
              "A small GPT learns day by day. Between days it 'sleeps': keeps its old predictions on its own dreams.",
              [("Normal day-by-day training", 0.58, False), ("+ sleep on its own dreams", 0.19, True),
               ("+ stronger sleep (keep weight 3, 2,048 dreams)", 0.08, True)], f2,
              "Forgetting in bits per byte, lower is better; 3 seeds per arm. The stronger setting was the best of "
              "4 pre-defined arms, picked after the run. Much less forgetting, but not zero."),
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
    order = [snaps[0], snaps[1], snaps[2], *[Image.open(p).convert("RGB") for p in charts], snaps[3]]
    order[0].save(OUT / "linkedin-carousel.pdf", save_all=True, append_images=order[1:], resolution=100)
    for i, s in enumerate(snaps):
        s.save(OUT / f"slide-{i + 1}.png")


if __name__ == "__main__":
    charts = render_charts()
    snaps = render_video()
    render_carousel(snaps, charts)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:40s} {p.stat().st_size / 1e6:6.2f} MB")
