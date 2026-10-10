"""Render logs/papers/*.md to paper-style PDFs with headless Chrome.
Run: python tools/md_to_pdf.py   (needs the `markdown` package and Google Chrome)."""
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "logs" / "papers"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CSS = """
@page { size: Letter; margin: 2.2cm 2.3cm; }
body { font-family: Cambria, Georgia, serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 6pt; line-height: 1.25; }
h1 + p { text-align: center; color: #444; margin-bottom: 16pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; font-size: 9pt; page-break-inside: avoid; }
th, td { border-top: 1px solid #bbb; border-bottom: 1px solid #bbb; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f2f2f0; }
code { font-family: Consolas, monospace; font-size: 9pt; }
pre { background: #f6f6f4; padding: 6pt; font-size: 8.5pt; white-space: pre-wrap; }
"""


def render(md_path):
    body = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    html = f"<!doctype html><meta charset='utf-8'><style>{CSS}</style><body>{body}</body>"
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html)
    pdf = md_path.with_suffix(".pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", Path(f.name).as_uri()], check=True, capture_output=True)
    Path(f.name).unlink()
    return pdf


if __name__ == "__main__":
    for p in sorted(SRC.glob("*.md")):
        print(render(p).name)
