"""Package exactly the manifest-listed slide previews into the presentation PDF."""

import json
from pathlib import Path

from reportlab.pdfgen.canvas import Canvas

root = Path("presentation")
manifest = json.loads((root / "slide_manifest.json").read_text(encoding="utf-8"))
slides = manifest["slides"]
assert slides and len({s["preview"] for s in slides}) == len(slides)
paths = [(root / s["preview"]).resolve() for s in slides]
for p in paths:
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError(f"Invalid slide preview: {p}")
canvas = Canvas(str(root / "presentation.pdf"), pagesize=(960, 540))
canvas.setTitle("CSC: metric development and evaluation on Iris and Breast Cancer")
for p in paths:
    canvas.drawImage(str(p), 0, 0, width=960, height=540)
    canvas.showPage()
canvas.save()
print(f"Packaged {len(paths)} manifest-listed slides.")
