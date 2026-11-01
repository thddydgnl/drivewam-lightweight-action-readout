"""Render every page of a PDF to PNG using only pypdf + macOS sips (no poppler needed)."""
import os, subprocess, sys, tempfile
from pypdf import PdfReader, PdfWriter
src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
r = PdfReader(src)
for i, page in enumerate(r.pages):
    w = PdfWriter(); w.add_page(page)
    tmp = os.path.join(outdir, f"p{i+1:02d}.pdf")
    with open(tmp, "wb") as f: w.write(f)
    subprocess.run(["sips", "-s", "format", "png", "-Z", "1400", tmp, "--out", os.path.join(outdir, f"p{i+1:02d}.png")], capture_output=True)
    os.remove(tmp)
print(len(r.pages), "pages ->", outdir)
